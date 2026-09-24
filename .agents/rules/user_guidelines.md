---
title: Reporting and Visualization Guidelines
trigger: model_decision
description: "Apply when creating or editing reports, presentation figures, plots, charts, validation graphics, or other visual material for the KABOAT/USV project."
---

# KABOAT Reporting and Visualization Guidelines

## Writing style

- Use a precise academic/engineering tone.
- Avoid decorative language, exaggerated claims, and emotional phrasing.
- In generated report text, do not use emoji.
- Avoid Markdown asterisk emphasis in report deliverables when the target format expects HTML-style emphasis; use headings, tables, or `<b>...</b>` when appropriate.
- Distinguish measured results, code-derived facts, engineering assumptions, and interpretation.

## Evidence

Do not present synthetic/example plots as measured simulation results.

Every performance claim should state the evaluation conditions when available:

- number of episodes;
- seed policy;
- map/environment conditions;
- timestep;
- planning frequency;
- relevant controller/dynamics version.

When comparing algorithms or revisions, use identical conditions whenever possible.

Success, collision, and timeout must be separated.

## Required vessel-focused visualizations

For performance reports or presentation material, prefer real run data and include a useful subset of:

- vessel trajectory;
- heading versus time;
- yaw rate / yaw acceleration;
- steering or differential-thrust command;
- linear speed;
- minimum obstacle clearance;
- path length / completion time;
- failure-category distribution;
- before/after comparison on identical seeds.

Do not generate all plots mechanically. Choose figures that support the engineering point being made.

## Figure quality

Before finalizing a figure, verify:

- labels do not overlap;
- legends do not obscure important data;
- boxes and annotations stay within bounds;
- axis units are explicit;
- text remains readable at presentation size;
- visual encodings are consistent across before/after figures.

Recommended minimum presentation sizes:

- annotations/body text: 10 pt;
- axis labels: 11 pt;
- subplot/section titles: 13 pt.

Use adequate contrast and avoid unnecessary visual clutter.

## Presentation use

For PPT or booth material:

- prefer one engineering message per figure;
- favor directly comparable before/after layouts;
- show the metric and its engineering implication together;
- avoid claiming real-vessel accuracy without measured real-vessel validation;
- label model-based or estimated quantities clearly.

## Git

Reporting work follows the Git rules in the root `AGENTS.md`.
Do not automatically commit or push merely because a report or figure was generated.
