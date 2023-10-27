---
title: Home
description: "QSym²: a program for Quantum Symbolic Symmetry"
icon: material/home
hide:
  - navigation
  - toc
---

<div class="qsym2_banner">
    <img src="assets/logos/qsym2_logo_no_background_light.svg#only-dark" alt="QSym² logo (light)">
    <img src="assets/logos/qsym2_logo_no_background_dark.svg#only-light" alt="QSym² logo (dark)">
    <p>
        A program for <b>Q</b>uantum <b>Sym</b>bolic <b>Sym</b>metry
    </p>
    <div class="container"; style="margin-bottom:3em;">
        <div class="row">
            <div class="col-sl-2 col-sl-2">
                <button class="emphasised" onclick="location.href='getting-started/prerequisites'">Get started</button>
                <button class="primary" onclick="location.href='#capabilities'">Learn more</button>
            </div>
        </div>
    </div>
</div>
<hr/>
<section id="capabilities">
    <div class="qsym2_heading">
        <h1>Capabilities</h1>
    </div>
    <div class="container"; style="margin-bottom:3em;">
        <div class="row">
            <div class="col-md-3 col-md-3 col-md-3">
                <div class="card_box">
                    <div onclick="location.href='methodologies/character-table-generation'" class="card">
                        <div class="card_title">
                            <div><span class="twemoji lg middle"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2m0 4v4h6V8H5m8 0v4h6V8h-6m-8 6v4h6v-4H5m8 0v4h6v-4h-6Z"/></svg></span> Character table generation</div>
                            <hr>
                        </div>
                        <div class="card_description">
                            <p>
                                Using the Burnside&ndash;Dixon algorithm, QSym²
                                can compute character tables symbolically on-the-fly for any finite group.
                            </p>
                        </div>
                    </div>
                    <div onclick="location.href='methodologies/orbit-based-representation-analysis'" class="card">
                        <div class="card_title">
                            <div><span class="twemoji lg middle"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M17 22v-2h3v-3h2v3.5c0 .39-.16.74-.46 1.04-.3.3-.65.46-1.04.46H17M7 22H3.5c-.39 0-.74-.16-1.04-.46-.3-.3-.46-.65-.46-1.04V17h2v3h3v2M17 2h3.5c.39 0 .74.16 1.04.46.3.3.46.65.46 1.04V7h-2V4h-3V2M7 2v2H4v3H2V3.5c0-.39.16-.74.46-1.04.3-.3.65-.46 1.04-.46H7m6 15.25 4-2.3v-4.59l-4 2.3v4.59m-1-6.33 4-2.29-4-2.35-4 2.35 4 2.29m-5 4.03 4 2.3v-4.59l-4-2.3v4.59m11.23-7.36c.5.32.77.75.77 1.32v6.32c0 .57-.27 1-.77 1.32l-5.48 3.18c-.5.32-1 .32-1.5 0l-5.48-3.18c-.5-.32-.77-.75-.77-1.32V8.91c0-.57.27-1 .77-1.32l5.48-3.18c.25-.13.5-.19.75-.19s.5.06.75.19l5.48 3.18Z"/></svg></span> Degeneracy detection</div>
                            <hr>
                        </div>
                        <div class="card_description">
                            <p>
                                With bespoke character tables, QSym² can handle degeneracy faithfully without recourse to Abelian subgroups.
                            </p>
                        </div>
                    </div>
                    <div onclick="location.href='methodologies/orbit-based-representation-analysis'" class="card">
                        <div class="card_title">
                            <div><span class="twemoji lg middle"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 448 512"><!--! Font Awesome Free 6.4.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free (Icons: CC BY 4.0, Fonts: SIL OFL 1.1, Code: MIT License) Copyright 2023 Fonticons, Inc.--><path d="M192 64v64c0 17.7 14.3 32 32 32h64c17.7 0 32-14.3 32-32V64c0-17.7-14.3-32-32-32h-64c-17.7 0-32 14.3-32 32zM82.7 207c-15.3 8.8-20.5 28.4-11.7 43.7l32 55.4c8.8 15.3 28.4 20.5 43.7 11.7l55.4-32c15.3-8.8 20.5-28.4 11.7-43.7l-32-55.4c-8.8-15.3-28.4-20.5-43.7-11.7l-55.4 32zM288 192c-17.7 0-32 14.3-32 32v64c0 17.7 14.3 32 32 32h64c17.7 0 32-14.3 32-32v-64c0-17.7-14.3-32-32-32h-64zm64 160c-17.7 0-32 14.3-32 32v64c0 17.7 14.3 32 32 32h64c17.7 0 32-14.3 32-32v-64c0-17.7-14.3-32-32-32h-64zm-192 32v64c0 17.7 14.3 32 32 32h64c17.7 0 32-14.3 32-32v-64c0-17.7-14.3-32-32-32h-64c-17.7 0-32 14.3-32 32zM32 352c-17.7 0-32 14.3-32 32v64c0 17.7 14.3 32 32 32h64c17.7 0 32-14.3 32-32v-64c0-17.7-14.3-32-32-32H32z"/></svg></span> Symmetry breaking analysis</div>
                            <hr>
                        </div>
                        <div class="card_description">
                            <p>
                                Via an orbit-based representation analysis method, QSym² can characterise the linear spans of symmetry-broken quantities.
                            </p>
                        </div>
                    </div>
                    <div onclick="location.href='methodologies/external-fields'" class="card">
                        <div class="card_title">
                            <div><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="m10 12 4 4-4 4v-3.1c-4.56-.46-8-2.48-8-4.9 0-2.42 3.44-4.44 8-4.9v1.99C6.55 9.43 4 10.6 4 12c0 1.4 2.55 2.57 6 2.91V12m10 0c0-1.4-2.55-2.57-6-2.91V7.1c4.56.46 8 2.48 8 4.9 0 2.16-2.74 4-6.58 4.7l.7-.7-1.2-1.21C17.89 14.36 20 13.27 20 12M11 2h2v11l-2-2V2m0 20v-1l2-2v3h-2Z"/></svg></span> Symmetry in external fields</div>
                            <hr>
                        </div>
                        <div class="card_description">
                            <p>
                                By means of fictitious special atoms, QSym² can perform symmetry analysis in the presence of electric and magnetic fields.
                            </p>
                        </div>
                    </div>
                    <div onclick="location.href='#'" class="card">
                        <div class="card_title">
                            <div><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" transform="rotate(30)"><path d="M19 4h1V1h-4v3s2 4 2 8-2 7-6 7-6-3-6-7 2-8 2-8V1H4v3h1S2 8 2 14c0 5 5 9 10 9s10-4 10-9c0-6-3-10-3-10M4 13c-.6 0-1-.4-1-1s.4-1 1-1 1 .4 1 1-.4 1-1 1m2 6c-.6 0-1-.4-1-1s.4-1 1-1 1 .4 1 1-.4 1-1 1m6 3c-.6 0-1-.4-1-1s.4-1 1-1 1 .4 1 1-.4 1-1 1m6-3c-.6 0-1-.4-1-1s.4-1 1-1 1 .4 1 1-.4 1-1 1m2-6c-.6 0-1-.4-1-1s.4-1 1-1 1 .4 1 1-.4 1-1 1Z"/></svg></span> Magnetic corepresentations</div>
                            <hr>
                        </div>
                        <div class="card_description">
                            <p>
                                Via a proper treatment of anti-unitary operators, QSym² can analyse magnetic symmetry using Wigner's corepresentation theory.
                            </p>
                        </div>
                    </div>
                    <div onclick="location.href='#'" class="card">
                        <div class="card_title">
                            <div><span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M21.807 18.285 13.553.756a1.324 1.324 0 0 0-1.129-.754 1.31 1.31 0 0 0-1.206.626l-8.952 14.5a1.356 1.356 0 0 0 .016 1.455l4.376 6.778a1.408 1.408 0 0 0 1.58.581l12.703-3.757c.389-.115.707-.39.873-.755s.164-.783-.007-1.145zm-1.848.752L9.18 22.224a.452.452 0 0 1-.575-.52l3.85-18.438c.072-.345.549-.4.699-.08l7.129 15.138a.515.515 0 0 1-.325.713z"/></svg></span> Projective (co)representations</div>
                            <hr>
                        </div>
                        <div class="card_description">
                            <p>
                                By explicit spin rotations, QSym² can handle projective (or double-valued) representations and corepresentations.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>

<hr>

<section id="adoptors">
    <div class="qsym2_heading">
        <h1>Adoptors</h1>
    </div>
    <div class="container"; style="margin-bottom:3em;">
        <div class="row">
            <div class="col-lg-2 col-lg-2">
                <div class="card_box">
                    <div onclick="window.open('https://quest.codes/', '_blank')" class="card">
                        <div class="card_title">
                            <div>QUEST</div>
                            <hr>
                        </div>
                        <div class="card_description">
                            <div div style="text-align: center;">
                                <img src="assets/logos/quest_logo_no_text_light.svg#only-dark" alt="QUEST logo (light)" id="quest-logo">
                                <img src="assets/logos/quest_logo_no_text_dark.svg#only-light" alt="QUEST logo (dark)" id="quest-logo">
                            </div>
                            <p>
                                QUEST employs QSym² to perform symmetry analysis for many of its electronic-structure calculations in the presence of external fields.
                            </p>
                        </div>
                    </div>
                    <div onclick="window.open('https://gitlab.com/Bspeake/questview', '_blank')" class="card">
                        <div class="card_title">
                            <div>QuestView</div>
                            <hr>
                        </div>
                        <div class="card_description">
                            <div div style="text-align: center;">
                                <img src="assets/logos/questview_logo_no_text.svg" alt="QuestView logo" id="questview-logo">
                            </div>
                            <p>
                                QuestView utilises QSym² as a backend to compute the unitary and anti-unitary symmetry elements that it visualises.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>

<!-- ## Commands -->
<!---->
<!-- * `mkdocs new [dir-name]` - Create a new project. -->
<!-- * `mkdocs serve` - Start the live-reloading docs server. -->
<!-- * `mkdocs build` - Build the documentation site. -->
<!-- * `mkdocs -h` - Print help message and exit. -->
<!---->
<!-- ## Project layout -->
<!---->
<!--     mkdocs.yml    # The configuration file. -->
<!--     docs/ -->
<!--         index.md  # The documentation homepage. -->
<!--         ...       # Other markdown pages, images and other files. -->
<!---->
<!-- ## Test code highlighting -->
<!---->
<!-- This is Rust: -->
<!-- ```rust linenums="1" hl_lines="2" -->
<!-- let a: Vec<u32> = vec![0, 1, 2]; // (1)! -->
<!-- let b = a.iter().map(|x| x * 2).collect::<Vec<_>>(); // (2)! -->
<!-- ``` -->
<!---->
<!-- 1. This is to define a *new* variable called `a`. -->
<!-- 2. This iterates over `a` and defines a new variable called `b`. -->
<!---->
<!-- ???+ warning "A short aside" -->
<!--     One must be careful when one uses Rust. -->
<!---->
<!-- These are different ways of achieving the same thing: -->
<!-- === "Rust" -->
<!--     ```rust -->
<!--     let a = vec![1, 2, 3]; -->
<!--     ``` -->
<!---->
<!-- === "Python" -->
<!--     ```python -->
<!--     a = [1, 2, 3] -->
<!--     ``` -->
<!---->
<!-- ## Test maths -->
<!---->
<!-- This is an inline equation: $E = mc^2$. -->
<!---->
<!-- This is a display equation: -->
<!---->
<!-- $$ -->
<!--     \int_{-\infty}^{\infty} \exp(-r^2) dr. -->
<!-- $$ -->
