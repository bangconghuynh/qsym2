use std::path::{Path, PathBuf};

use clap::Parser;

use crate::io::format::qsym2_output;

const VERSION: Option<&str> = option_env!("CARGO_PKG_VERSION");

/// Logs a nicely formatted QSym2 heading to the `qsym2-output` logger.
pub fn log_heading() {
    let version = if let Some(ver) = VERSION {
        format!("v{ver}")
    } else {
        format!("v unknown")
    };
    qsym2_output!("╭─────────────────────────────────────────────────────────────────────────────────────────────────────╮");
    qsym2_output!("│                                                                                 222222222222222     │");
    qsym2_output!("│                                                                                2:::::::::::::::22   │");
    qsym2_output!("│                                                                                2::::::222222:::::2  │");
    qsym2_output!("│                                                                                2222222     2:::::2  │");
    qsym2_output!("│                                                                                            2:::::2  │");
    qsym2_output!("│      QQQQQQQQQ        SSSSSSSSSSSSSSS                                                      2:::::2  │");
    qsym2_output!("│    QQ:::::::::QQ    SS:::::::::::::::S                                                  2222::::2   │");
    qsym2_output!("│  QQ:::::::::::::QQ S:::::SSSSSS::::::S                                             22222::::::22    │");
    qsym2_output!("│ Q:::::::QQQ:::::::QS:::::S     SSSSSSS                                           22::::::::222      │");
    qsym2_output!("│ Q::::::O   Q::::::QS:::::S      yyyyyyy           yyyyyyy mmmmmmm    mmmmmmm    2:::::22222         │");
    qsym2_output!("│ Q:::::O     Q:::::QS:::::S       y:::::y         y:::::ymm:::::::m  m:::::::mm 2:::::2              │");
    qsym2_output!("│ Q:::::O     Q:::::Q S::::SSSS     y:::::y       y:::::ym::::::::::mm::::::::::m2:::::2              │");
    qsym2_output!("│ Q:::::O     Q:::::Q  SS::::::SSSSS y:::::y     y:::::y m::::::::::::::::::::::m2:::::2       222222 │");
    qsym2_output!("│ Q:::::O     Q:::::Q    SSS::::::::SSy:::::y   y:::::y  m:::::mmm::::::mmm:::::m2::::::2222222:::::2 │");
    qsym2_output!("│ Q:::::O     Q:::::Q       SSSSSS::::Sy:::::y y:::::y   m::::m   m::::m   m::::m2::::::::::::::::::2 │");
    qsym2_output!("│ Q:::::O  QQQQ:::::Q            S:::::Sy:::::y:::::y    m::::m   m::::m   m::::m22222222222222222222 │");
    qsym2_output!("│ Q::::::O Q::::::::Q            S:::::S y:::::::::y     m::::m   m::::m   m::::m                     │");
    qsym2_output!("│ Q:::::::QQ::::::::QSSSSSSS     S:::::S  y:::::::y      m::::m   m::::m   m::::m                     │");
    qsym2_output!("│  QQ::::::::::::::Q S::::::SSSSSS:::::S   y:::::y       m::::m   m::::m   m::::m                     │");
    qsym2_output!("│    QQ:::::::::::Q  S:::::::::::::::SS   y:::::y        m::::m   m::::m   m::::m                     │");
    qsym2_output!("│      QQQQQQQQ::::QQ SSSSSSSSSSSSSSS    y:::::y         mmmmmm   mmmmmm   mmmmmm                     │");
    qsym2_output!("│              Q:::::Q                  y:::::y                                                       │");
    qsym2_output!("│               QQQQQQ                 y:::::y                                                        │");
    qsym2_output!("│                                     y:::::y                                                         │");
    qsym2_output!("│                                    y:::::y                                            {version:>13} │");
    qsym2_output!("│                                   yyyyyyy                                     Author: Bang C. Huynh │");
    qsym2_output!("╰─────────────────────────────────────────────────────────────────────────────────────────────────────╯");
    qsym2_output!("");
}

#[derive(Parser)]
#[command(author, version, about)]
pub struct Cli {
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    #[arg(short, long)]
    pub output: Option<PathBuf>,
}
