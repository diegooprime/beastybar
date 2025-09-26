# Developer Notes

## Card behaviour assumptions
- Seal swaps the entire contents of `beasty_bar` and `bounced` immediately on play; any subsequent recurring effects (e.g., crocodile) operate on the swapped zones as usual.
- Chameleon requires the index of the card it copies. Extra parameters after the first are forwarded to the copied species (currently only the parrot needs a target index). If the target card accepts no params, the chameleon action must omit them.
- Parrot and chameleon-targeted parrots always require a single queue index parameter referencing a card currently in line.
- Recurring abilities resolve after the on-play effect, scanning from gate to bounce: hippos pass first, then crocodiles eat, then giraffes hop. Zebras block both hippo passes and crocodile bites for themselves and animals ahead of them.
- The default batch-agent policy picks the first legal action. Tests should use custom agents (like the scripted helper in `tests/test_simulate.py`) when deterministic behaviour is required.

## Testing reminders
- Run `python3 -m pytest` before committing; this covers golden replays, card behaviours, action validation, and batch API scaffolding.
- Update `tests/test_replay.py` or add new fixtures when introducing card logic that alters turn sequencing so regression coverage stays accurate.
