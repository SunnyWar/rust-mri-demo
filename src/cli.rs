use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    pub(crate) root: String,

    #[arg(short = 's', long = "sigma-3d", default_value_t = 1.5)]
    pub(crate) sigma_3d: f32,

    #[arg(long = "sigma-fa", default_value_t = 1.0)]
    pub(crate) sigma_fa: f32,

    #[arg(short, long, default_value_t = 1000.0)]
    pub(crate) bvalue: f32,

    /// Also output a mean diffusivity map
    #[arg(long)]
    pub(crate) emit_md: bool,

    /// Also output an axial diffusivity map
    #[arg(long)]
    pub(crate) emit_ad: bool,

    /// Also output a radial diffusivity map
    #[arg(long)]
    pub(crate) emit_rd: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_defaults() {
        let args = Cli::try_parse_from(["dti_pipeline", "/path/to/data"]).unwrap();
        assert_eq!(args.root, "/path/to/data");
        assert_eq!(args.sigma_3d, 1.5);
        assert_eq!(args.sigma_fa, 1.0);
        assert_eq!(args.bvalue, 1000.0);
        assert!(!args.emit_md);
        assert!(!args.emit_ad);
        assert!(!args.emit_rd);
    }

    #[test]
    fn test_cli_custom_flags() {
        let args = Cli::try_parse_from([
            "dti_pipeline",
            "/path/to/data",
            "-s",
            "2.5",
            "--sigma-fa",
            "0.8",
            "-b",
            "1500.0",
            "--emit-md",
            "--emit-rd",
        ])
        .unwrap();

        assert_eq!(args.sigma_3d, 2.5);
        assert_eq!(args.sigma_fa, 0.8);
        assert_eq!(args.bvalue, 1500.0);
        assert!(args.emit_md);
        assert!(!args.emit_ad);
        assert!(args.emit_rd);
    }
}
