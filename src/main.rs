use std::path::PathBuf;

use clap::Parser;
use log;
use log::LevelFilter;
use log4rs::append::console::ConsoleAppender;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Logger, Root};
use log4rs::encode::pattern::PatternEncoder;

use qsym2::interfaces::cli::{
    qsym2_output_calculation_summary, qsym2_output_contributors, qsym2_output_heading, Cli, Commands
};
use qsym2::io::{read_qsym2_yaml, write_qsym2_yaml};

use qsym2::interfaces::input::Input;
use qsym2::interfaces::InputHandle;

fn try_main() -> Result<(), anyhow::Error> {
    // Parse CLI arguments
    let cli = Cli::parse();

    match &cli.command {
        Commands::Template { name } => {
            let name = name.clone().unwrap_or(PathBuf::from("template_config.yml"));
            let default_input = Input::default();
            write_qsym2_yaml(name, &default_input)
        }
        Commands::Run { config, output, debug } => {
            let config_path = config;
            let output_path = output;
            let mut debug_path = output_path.to_path_buf();
            debug_path.set_extension("dbg");

            // Parse input config
            let input_config = read_qsym2_yaml::<Input, _>(config_path)?;

            // Set up loggers
            let stdout = ConsoleAppender::builder()
                .encoder(Box::new(PatternEncoder::new(
                    "{d(%Y-%m-%d %H:%M:%S %Z)(utc)} {h({l:<5})} {t} - {m}{n}",
                )))
                .build();

            let output_log_appender = FileAppender::builder()
                .encoder(Box::new(PatternEncoder::new("{m}{n}")))
                .append(false)
                .build(output_path)?;

            match debug {
                0 => {
                    // Main output to output file
                    // Warnings and errors to stdout
                    let output_log_config = Config::builder()
                        .appender(Appender::builder().build("stdout", Box::new(stdout)))
                        .appender(Appender::builder().build("output_ap", Box::new(output_log_appender)))
                        .logger(
                            Logger::builder()
                                .appender("output_ap")
                                .additive(false)
                                .build("qsym2-output", LevelFilter::Info),
                        )
                        .build(Root::builder().appender("stdout").build(LevelFilter::Warn))?;
                    log4rs::init_config(output_log_config)?;
                }
                1 => {
                    // Main output to output file
                    // Debugs, main output, warnings and errors to debug file
                    // Non-qsym2 warnings and errors to stdout
                    let debug_log_appender = FileAppender::builder()
                        .encoder(Box::new(PatternEncoder::new(
                            "{d(%Y-%m-%d %H:%M:%S %Z)(utc)} {h({l:<5})} {t} - {m}{n}",
                        )))
                        .append(false)
                        .build(debug_path)?;
                    let output_log_config = Config::builder()
                        .appender(Appender::builder().build("stdout", Box::new(stdout)))
                        .appender(Appender::builder().build("output_ap", Box::new(output_log_appender)))
                        .appender(Appender::builder().build("debug_ap", Box::new(debug_log_appender)))
                        .logger(
                            Logger::builder()
                                .appender("output_ap")
                                .appender("debug_ap")
                                .additive(false)
                                .build("qsym2-output", LevelFilter::Info),
                        )
                        .logger(
                            Logger::builder()
                                .appender("debug_ap")
                                .additive(false)
                                .build("qsym2", LevelFilter::Debug),
                        )
                        .build(Root::builder().appender("stdout").build(LevelFilter::Warn))?;
                    log4rs::init_config(output_log_config)?;
                }
                _ => {
                    // Main output to output file
                    // All debugs, main output, warnings and errors to debug file and stdout
                    let debug_log_appender = FileAppender::builder()
                        .encoder(Box::new(PatternEncoder::new(
                            "{d(%Y-%m-%d %H:%M:%S %Z)(utc)} {h({l:<5})} {t} - {m}{n}",
                        )))
                        .append(false)
                        .build(debug_path)?;
                    let output_log_config = Config::builder()
                        .appender(Appender::builder().build("stdout", Box::new(stdout)))
                        .appender(Appender::builder().build("output_ap", Box::new(output_log_appender)))
                        .appender(Appender::builder().build("debug_ap", Box::new(debug_log_appender)))
                        .logger(
                            Logger::builder()
                                .appender("output_ap")
                                .additive(true)
                                .build("qsym2-output", LevelFilter::Info),
                        )
                        .build(
                            Root::builder()
                                .appender("debug_ap")
                                .appender("stdout")
                                .build(LevelFilter::Debug),
                        )?;
                    log4rs::init_config(output_log_config)?;
                }
            };

            qsym2_output_heading();
            qsym2_output_contributors();
            qsym2_output_calculation_summary(config_path, &cli);

            input_config.handle()
        }
    }
}

fn main() -> Result<(), anyhow::Error> {
    let main_res = try_main();
    if let Err(err) = main_res.as_ref() {
        log::error!("{err:#}");
    }
    main_res
}
