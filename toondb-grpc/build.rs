// Build script for toondb-grpc
// Compiles protobuf definitions using tonic-build

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the output directory from cargo
    let out_dir = std::env::var("OUT_DIR")?;
    
    // Compile the proto file
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(&out_dir)
        .compile_protos(
            &["proto/toondb.proto"],
            &["proto"],
        )?;
    
    println!("cargo:rerun-if-changed=proto/toondb.proto");
    Ok(())
}
