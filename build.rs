use gl_generator::{Api, Fallbacks, Profile, Registry, StructGenerator};
use std::env;
use std::fs::File;
use std::path::PathBuf;

fn main() {
    let dest = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut file = File::create(dest.join("gl_bindings.rs")).unwrap();

    // Compatibility profile so the legacy fixed-function pipeline functions
    // (glBegin/glEnd, glFrustum, glLight*, glMaterial*, ...) used by the pulsar
    // scene are available, exactly like the original C/GLUT demo.
    Registry::new(Api::Gl, (4, 5), Profile::Compatibility, Fallbacks::All, [])
        .write_bindings(StructGenerator, &mut file)
        .unwrap();
}
