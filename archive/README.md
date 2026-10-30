# Archive

Superseded scripts, kept for reference. **They are not runnable.**

They predate the `src/transformer2/` package and import modules by their old flat
names (`coreTransformer`, `dataloader`, `trainingSetup`), which no longer exist.
Nothing in the live codebase imports from here.

Their successors: `trainingSetup.py`/`curriculumTraining.py` → `training/trainer.py`,
`getRedditTraining.py`/`getBlueskyTraining.py` → `data/collection.py`,
`inferenceTest.py` → `inference/generate.py`.
