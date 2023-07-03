use clap::Parser;
use log;
use log::LevelFilter;
use log4rs::append::console::ConsoleAppender;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Logger, Root};
use log4rs::encode::pattern::PatternEncoder;

use qsym2::interfaces::cli::{log_heading, Cli};
use qsym2::drivers::representation_analysis::CharacterTableDisplay;
use qsym2::io::read_qsym2_yaml;

use qsym2::interfaces::input::Input;

fn main() {
    // Parse CLI arguments
    let cli = Cli::parse();
    let config_path = cli
        .config
        .as_deref()
        .expect("No configuration file specified with -c/--config.");
    let output_path = cli
        .output
        .as_deref()
        .expect("No output file specified with -o/--output.");

    // Parse input config
    let config = read_qsym2_yaml::<Input, _>(config_path).unwrap_or_else(|err| {
        log::error!("{err}");
        panic!("Failed to parse the configuration file with error: {err}");
    });

    // Set up loggers
    let stdout = ConsoleAppender::builder().build();

    let output_log_appender = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{m}{n}")))
        .append(false)
        .build(output_path)
        .expect("Unable to construct an output log `FileAppender`.");

    let output_log_config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .appender(Appender::builder().build("output_ap", Box::new(output_log_appender)))
        .logger(
            Logger::builder()
                .appender("output_ap")
                .additive(false)
                .build("qsym2-output", LevelFilter::Info),
        )
        .build(Root::builder().appender("stdout").build(LevelFilter::Warn))
        .expect("Unable to construct an output log `Config`.");

    let handle = log4rs::init_config(output_log_config).unwrap();

    log_heading();
}
