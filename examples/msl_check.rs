//! Compiles the Metal kernel libraries and prints any compile errors; uses the Metal compiler only, no GPU time

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("the Metal backend is macOS only");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
fn main() {
    let libs = match teasar::metal::gpu::check_libraries() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("no Metal device: {e}");
            std::process::exit(2);
        }
    };
    let mut failed = false;
    for (name, err) in libs {
        match err {
            None => println!("{name}: ok"),
            Some(e) => {
                failed = true;
                println!("{name}: FAILED\n{e}");
            }
        }
    }
    std::process::exit(i32::from(failed));
}
