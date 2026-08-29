//! Modern-GL rendering helpers for the demo.
//!
//! The original fixed-function pipeline (immediate mode + display lists,
//! lighting, texture environment) is replaced by two tiny GLSL programs -
//! flat colour and textured - backed by vertex buffer objects. `glow`'s API
//! is `unsafe`, and GL errors are process-fatal by design, so every `glow`
//! call in the demo lives here behind these narrow, safe wrappers (render
//! scenes never touch `glow` directly).
//!
//! `bytemuck` provides the vertex uploads: the vertex structs are
//! byte-layout-stable (`Pod`/`Zeroable`) and are uploaded with
//! `buffer_data_u8_slice`, keeping the data path allocation-free per frame.

use crate::gl::{self, HasContext};
use crate::Gl;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Vertex formats
// ---------------------------------------------------------------------------

/// Position + RGB colour vertex (scene geometry).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ColorVert {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

/// 2D position + UV vertex (textured quads).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct UvVert {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

const COLOR_VS: &str = r#"#version 120
attribute vec3 a_pos;
attribute vec3 a_color;
uniform mat4 u_mvp;
varying vec3 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}"#;

const COLOR_FS: &str = r#"#version 120
varying vec3 v_color;
void main() {
    gl_FragColor = vec4(v_color, 1.0);
}"#;

const TEX_VS: &str = r#"#version 120
attribute vec2 a_pos;
attribute vec2 a_uv;
uniform mat4 u_mvp;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = u_mvp * vec4(a_pos, 0.0, 1.0);
}"#;

const TEX_FS: &str = r#"#version 120
uniform sampler2D u_tex;
uniform vec4 u_color;
varying vec2 v_uv;
void main() {
    gl_FragColor = texture2D(u_tex, v_uv) * u_color;
}"#;

/// A compiled+linked program with its uniforms resolved. `Copy` (wraps GL
/// object names), so it can be cached in a `thread_local!` and handed out by
/// value.
#[derive(Clone, Copy)]
pub struct Program {
    id: Option<gl::Program>,
    u_mvp: Option<gl::UniformLocation>,
    u_color: Option<gl::UniformLocation>,
}

pub const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

impl Program {
    fn compile(gl: &Gl, vs: &str, fs: &str) -> Option<Program> {
        unsafe {
            let vert = gl.create_shader(gl::VERTEX_SHADER).ok()?;
            gl.shader_source(vert, vs);
            gl.compile_shader(vert);
            if !gl.get_shader_compile_status(vert) {
                eprintln!("gfx: vertex shader compile failed: {}", gl.get_shader_info_log(vert));
                gl.delete_shader(vert);
                return None;
            }
            let frag = gl.create_shader(gl::FRAGMENT_SHADER).ok()?;
            gl.shader_source(frag, fs);
            gl.compile_shader(frag);
            if !gl.get_shader_compile_status(frag) {
                eprintln!("gfx: fragment shader compile failed: {}", gl.get_shader_info_log(frag));
                gl.delete_shader(vert);
                gl.delete_shader(frag);
                return None;
            }
            let prog = gl.create_program().ok()?;
            gl.attach_shader(prog, vert);
            gl.attach_shader(prog, frag);
            gl.link_program(prog);
            if !gl.get_program_link_status(prog) {
                eprintln!("gfx: program link failed: {}", gl.get_program_info_log(prog));
                gl.delete_program(prog);
                gl.delete_shader(vert);
                gl.delete_shader(frag);
                return None;
            }
            gl.delete_shader(vert);
            gl.delete_shader(frag);
            let u_mvp = gl.get_uniform_location(prog, "u_mvp");
            let u_color = gl.get_uniform_location(prog, "u_color");
            // The sampler always lives on texture unit 0.
            if let Some(u_tex) = gl.get_uniform_location(prog, "u_tex") {
                gl.use_program(Some(prog));
                gl.uniform_1_i32(Some(&u_tex), 0);
                gl.use_program(None);
            }
            Some(Program { id: Some(prog), u_mvp, u_color })
        }
    }

    /// Binds this program and sets its uniforms; the caller draws afterwards.
    fn bind(&self, gl: &Gl, mvp: Mat4, color: [f32; 4]) {
        unsafe {
            gl.use_program(self.id);
            if let Some(loc) = self.u_mvp {
                let cols = mvp.to_cols_array();
                gl.uniform_matrix_4_f32_slice(Some(&loc), false, &cols);
            }
            if let Some(loc) = self.u_color {
                gl.uniform_4_f32(Some(&loc), color[0], color[1], color[2], color[3]);
            }
        }
    }
}

/// The two canvas programs (flat-colour and textured), compiled once per GL
/// context. The demo has exactly one context per process (main thread), so a
/// `thread_local!` cache is the natural home, mirroring the display-list /
/// texture caches it replaces.
#[derive(Clone, Copy)]
pub struct Shaders {
    pub color: Program,
    pub tex: Program,
}

thread_local! {
    static SHADERS: RefCell<Option<Shaders>> = const { RefCell::new(None) };
}

/// Compiles (once) and returns the shared programs, or `None` if a program
/// failed to build (the scene then simply draws nothing, like a failed
/// display list did).
pub fn shaders(gl: &Gl) -> Option<Shaders> {
    SHADERS.with(|c| {
        let mut c = c.borrow_mut();
        if c.is_none() {
            let color = Program::compile(gl, COLOR_VS, COLOR_FS)?;
            let tex = Program::compile(gl, TEX_VS, TEX_FS)?;
            *c = Some(Shaders { color, tex });
        }
        *c
    })
}

// ---------------------------------------------------------------------------
// Meshes (VBOs)
// ---------------------------------------------------------------------------

/// Static vertex data in one VBO, drawn as one or more primitive ranges.
///
/// `replace_*_verts` re-uploads into an existing buffer (orphaning), so the
/// per-frame text/RDS quads do not create GL objects every frame.
pub struct Mesh {
    vbo: Option<gl::Buffer>,
    /// (first vertex, vertex count) of each primitive to draw.
    ranges: Vec<(i32, i32)>,
    mode: u32,
    /// (attrib index, components) for each source of this mesh's vertex type.
    layout: [(u32, i32); 2],
    stride: i32,
}

impl Mesh {
    fn build(gl: &Gl, data: &[u8], ranges: &[(i32, i32)], mode: u32, layout: [(u32, i32); 2], stride: i32) -> Mesh {
        let mut mesh = Mesh {
            vbo: None,
            ranges: ranges.to_vec(),
            mode,
            layout,
            stride,
        };
        unsafe {
            if let Ok(vbo) = gl.create_buffer() {
                gl.bind_buffer(gl::ARRAY_BUFFER, Some(vbo));
                gl.buffer_data_u8_slice(gl::ARRAY_BUFFER, data, gl::STATIC_DRAW);
                gl.bind_buffer(gl::ARRAY_BUFFER, None);
                mesh.vbo = Some(vbo);
            }
        }
        mesh
    }

    /// A flat-colour triangle mesh from position+colour vertices.
    pub fn from_color_verts(gl: &Gl, verts: &[ColorVert], mode: u32) -> Mesh {
        Mesh::build(
            gl,
            bytemuck::cast_slice(verts),
            &[(0, verts.len() as i32)],
            mode,
            [(0, 3), (1, 3)],
            24,
        )
    }

    /// A flat-colour mesh drawn as several distinct primitive ranges (the
    /// pulsar's field-line strips share one VBO).
    pub fn from_color_verts_ranges(gl: &Gl, verts: &[ColorVert], ranges: &[(i32, i32)], mode: u32) -> Mesh {
        Mesh::build(
            gl,
            bytemuck::cast_slice(verts),
            ranges,
            mode,
            [(0, 3), (1, 3)],
            24,
        )
    }

    /// A textured quad mesh from 2D position + UV vertices.
    pub fn from_uv_verts(gl: &Gl, verts: &[UvVert], mode: u32) -> Mesh {
        Mesh::build(
            gl,
            bytemuck::cast_slice(verts),
            &[(0, verts.len() as i32)],
            mode,
            [(0, 2), (1, 2)],
            16,
        )
    }

    /// Replaces this mesh's contents with new UV vertices, keeping the same
    /// VBO (orphans it so the driver can retire the old storage async).
    pub fn replace_uv_verts(&mut self, gl: &Gl, verts: &[UvVert]) {
        if let Some(vbo) = self.vbo {
            unsafe {
                gl.bind_buffer(gl::ARRAY_BUFFER, Some(vbo));
                gl.buffer_data_u8_slice(gl::ARRAY_BUFFER, bytemuck::cast_slice(verts), gl::STREAM_DRAW);
                gl.bind_buffer(gl::ARRAY_BUFFER, None);
            }
            self.ranges.clear();
            self.ranges.push((0, verts.len() as i32));
        }
    }

    fn draw_with(&self, gl: &Gl, bind: impl FnOnce()) {
        let vbo = match self.vbo {
            Some(vbo) => vbo,
            None => return,
        };
        unsafe {
            gl.bind_buffer(gl::ARRAY_BUFFER, Some(vbo));
            let mut offset = 0i32;
            for &(loc, comps) in &self.layout {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, comps, gl::FLOAT, false, self.stride, offset);
                offset += comps * 4;
            }
        }
        bind();
        unsafe {
            for &(first, count) in &self.ranges {
                gl.draw_arrays(self.mode, first, count);
            }
            for &(loc, _) in &self.layout {
                gl.disable_vertex_attrib_array(loc);
            }
            gl.bind_buffer(gl::ARRAY_BUFFER, None);
            gl.use_program(None);
        }
    }

    /// Draws all ranges with the flat-colour program (colour comes from the
    /// per-vertex attribute; the tint uniform stays white).
    pub fn draw_color(&self, gl: &Gl, mvp: Mat4) {
        let programs = match shaders(gl) {
            Some(p) => p,
            None => return,
        };
        self.draw_with(gl, || programs.color.bind(gl, mvp, WHITE));
    }

    /// Draws all ranges with the textured program. The texture must already
    /// be bound on unit 0; `color` modulates the sampled texel (white =
    /// textured-aligned, the `REPLACE` behaviour the original scenes used).
    pub fn draw_textured(&self, gl: &Gl, mvp: Mat4, color: [f32; 4]) {
        let programs = match shaders(gl) {
            Some(p) => p,
            None => return,
        };
        self.draw_with(gl, || programs.tex.bind(gl, mvp, color));
    }
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/// A 2D texture owned by the caller-built scene. `Copy` (wraps a GL object
/// name), so cached textures can be handed around by value.
#[derive(Clone, Copy)]
pub struct Texture {
    id: Option<gl::Texture>,
}

impl Texture {
    pub fn none() -> Texture {
        Texture { id: None }
    }

    pub fn is_none(&self) -> bool {
        self.id.is_none()
    }

    /// Creates a texture and uploads `data` (or reserves storage when
    /// `None`). `internal` is the internal format (e.g. `GL_RGBA`,
    /// `GL_LUMINANCE`), `format`/`ty` describe the upload.
    pub fn new(
        gl: &Gl,
        width: i32,
        height: i32,
        internal: i32,
        format: u32,
        ty: u32,
        data: Option<&[u8]>,
        filter: i32,
        wrap: i32,
    ) -> Texture {
        use gl::PixelUnpackData;
        unsafe {
            let id = match gl.create_texture() {
                Ok(id) => id,
                Err(_) => return Texture { id: None },
            };
            gl.bind_texture(gl::TEXTURE_2D, Some(id));
            gl.pixel_store_i32(gl::UNPACK_ALIGNMENT, 1);
            gl.tex_parameter_i32(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, filter);
            gl.tex_parameter_i32(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, filter);
            gl.tex_parameter_i32(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, wrap);
            gl.tex_parameter_i32(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, wrap);
            gl.tex_image_2d(
                gl::TEXTURE_2D,
                0,
                internal,
                width,
                height,
                0,
                format,
                ty,
                PixelUnpackData::Slice(data),
            );
            gl.bind_texture(gl::TEXTURE_2D, None);
            Texture { id: Some(id) }
        }
    }

    /// Uploads a pixel rectangle into an existing texture (the text overlay
    /// does this per line).
    pub fn sub_image(&self, gl: &Gl, x: i32, y: i32, width: i32, height: i32, format: u32, ty: u32, data: &[u8]) {
        use gl::PixelUnpackData;
        if let Some(id) = self.id {
            unsafe {
                gl.bind_texture(gl::TEXTURE_2D, Some(id));
                gl.pixel_store_i32(gl::UNPACK_ALIGNMENT, 1);
                gl.tex_sub_image_2d(
                    gl::TEXTURE_2D,
                    0,
                    x,
                    y,
                    width,
                    height,
                    format,
                    ty,
                    PixelUnpackData::Slice(Some(data)),
                );
                gl.bind_texture(gl::TEXTURE_2D, None);
            }
        }
    }

    /// Deletes the underlying GL texture. Used when a cached texture is
    /// replaced (e.g. the RDS base field on a surface resize).
    pub fn delete(&self, gl: &Gl) {
        if let Some(id) = self.id {
            unsafe { gl.delete_texture(id) }
        }
    }

    /// Binds the texture to unit 0.
    pub fn bind(&self, gl: &Gl) {
        unsafe { gl.bind_texture(gl::TEXTURE_2D, self.id) }
    }

    pub fn unbind(gl: &Gl) {
        unsafe { gl.bind_texture(gl::TEXTURE_2D, None) }
    }

    /// Binds, draws one mesh with the textured program, unbinds.
    pub fn draw(&self, gl: &Gl, mesh: &Mesh, mvp: Mat4, color: [f32; 4]) {
        self.bind(gl);
        mesh.draw_textured(gl, mvp, color);
        Texture::unbind(gl);
    }
}

// ---------------------------------------------------------------------------
// Global GL state helpers
// ---------------------------------------------------------------------------

pub fn clear(gl: &Gl, mask: u32) {
    unsafe { gl.clear(mask) }
}

pub fn clear_color(gl: &Gl, r: f32, g: f32, b: f32, a: f32) {
    unsafe { gl.clear_color(r, g, b, a) }
}

pub fn enable(gl: &Gl, cap: u32) {
    unsafe { gl.enable(cap) }
}

pub fn disable(gl: &Gl, cap: u32) {
    unsafe { gl.disable(cap) }
}

pub fn blend_func(gl: &Gl, src: u32, dst: u32) {
    unsafe { gl.blend_func(src, dst) }
}

pub fn viewport(gl: &Gl, x: i32, y: i32, w: i32, h: i32) {
    unsafe { gl.viewport(x, y, w, h) }
}