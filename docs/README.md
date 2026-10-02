# Documentation index

Start at the repository [README](../README.md) for setup and the project map.
Run FR3 commands from `fr3_real/` unless a guide says otherwise.

| Document | Use |
| --- | --- |
| [Bracket, wall-cell, and bolt workflows](BRACKET_BOLT_WORKFLOWS.md) | Current task boundaries, source data, loop commands, review and conversion. |
| [FR3 Script Guide](SCRIPT_GUIDE.md) | Operational entry points and their role. |
| [FR3 Recording Guide](RECORDING_GUIDE.md) | Released cube-to-basket recording workflow only. |
| [Project handoff](../fr3_real/PROJECT_HANDOFF.md) | Detailed August bracket/bolt commissioning history, not current cell status. |
| [Calibration README](../fr3_real/calibration/README.md) | Workcell-specific measurement and expert-pose records. |
| [Bracket training handoff](../fr3_real/deploy/bracket_training/BRACKET_TRAINING_HANDOFF.md) | Dated bracket policy training run. |
| [Bracket evaluation guide](../fr3_real/deploy/bracket_eval/BRACKET_EVAL_README.md) | Dated bracket policy evaluation run. |
| [FR3 research progress PDF](FR3_PROJECT_UPDATE_FOR_HUYUE_AND_DR_MEI.pdf) | Two-page report for Huyue and Dr Mei. |

The two `FR3_VLA_*.docx` files are paper drafts. They are not evaluation
results or evidence of publication. The editable source for the project
update PDF is `FR3_PROJECT_UPDATE_FOR_HUYUE_AND_DR_MEI.html`; regenerate its
DOCX with `python docs/build_project_update_docx.py` from the repository root,
then export that DOCX to PDF with LibreOffice. The DOCX requests Times New
Roman at 12 points. On systems without that font, the PDF exporter substitutes
a compatible serif font; inspect the embedded PDF font before sending if the
exact typeface is required.
