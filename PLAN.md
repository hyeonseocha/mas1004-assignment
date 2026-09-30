# The plan

This is the plan I would write before starting this assignment. Give it to your
coding agent one step at a time, not all at once. An agent handed twenty steps
will do a bad job of all twenty.

Read it first. Part of what this course is teaching is how to write one of
these yourself, and you cannot learn that by pasting it without looking.

Each step says what to do, and how you know it worked. The second part matters
more. A step is not done because the agent says it is done. It is done when the
check passes.

---

## Step 0. Make sure the room is empty before you move in

Run `uv run pytest tests/test_export.py`. Six tests should pass. They cover
code you were given, so if they fail, nothing you write afterwards will work
either.

Check: six passed.

## Step 1. Choose your categories

Not a coding step. Write down 3 to 5 categories and one sentence each on why
you chose them. Keep the sentences. They go in your report.

Check: you can say out loud what visible difference the model is supposed to
find between them, and you know where you will get images of them that cannot
be among your downloads (Step 4).

## Step 2. Download the images

```
uv run python src/collect.py --classes "your,categories,here" --n 150
cp -r data/raw data/clean
```

Check: `data/clean/` has one folder per category and each has at least 50 files
in it. If one category came back nearly empty, the search phrase is wrong.
Change the phrase, not the category.

## Step 3. Look at what you downloaded

```
uv run python src/clean.py look
```

Open every sheet in `results/cleaning/look/` and look at every picture. Do not
remove anything yet. Your first training run in Step 9 is on the data as it
came, so that Problem 4 has something to compare against.

Check: you can name three kinds of junk that came back.

## Step 4. Collect images from a new source

At least five per category, into `data/new_images/<category>/`, with the same
folder names as `data/clean`. They must come from a source you can be sure is
not in your downloads: your own photos, a friend's photos, frames from a video
you recorded. Not another web search. Save them as JPEG or PNG, not the HEIC
files an iPhone makes by default (README, Problem 5). If you take photos
yourself, take them in different places and different light. If you take them
all on the same desk in one evening, Problems 4 and 5 have nothing to say.

```
uv run python src/clean.py overlap
```

Check: every category has a folder in `data/new_images/` with at least five
images in it, `clean.py overlap` finds no near copies, and you can say in one
sentence why none of them can be among your downloads.

## Step 5. Write `prepare_image`

Give the agent the docstring of `prepare_image` in `src/data.py` and ask it to
write the body. Then run:

```
uv run pytest tests/test_data.py -k prepare
```

The tests will fail in specific ways. Give the agent the failure text as it is,
not your summary of it.

Check: every prepare test is green, and you can say what happens to a photo
that is not square.

## Step 6. Write `load_folder` and `split_train_test`

Same again, then `uv run pytest tests/test_data.py`.

Before you move on, read the test called
`test_split_has_no_image_on_both_sides` and make sure you understand why it
exists. If you do not understand why an image being in both halves is a
problem, ask the agent to explain it, and do not accept an answer you cannot
repeat.

Check: all of `tests/test_data.py` is green.

## Step 7. Write `build_model` and `train`

```
uv run pytest tests/test_train.py
```

The first run downloads the ImageNet weights, about 45 MB. The last test trains
on three easy blobs and expects over 90%. If your training loop has a bug, that
is where it shows up. Common ones: forgetting `optimiser.step()`, forgetting
`optimiser.zero_grad()`, training on the test set, measuring accuracy as a
percentage instead of a share, and moving the model to the GPU but not the
batch ("Expected all tensors to be on the same device").

Check: all of `tests/test_train.py` is green.

## Step 8. Write `predict_logits`, `accuracy` and `confusion_matrix`

```
uv run pytest tests/test_evaluate.py -k "not worst"
```

Check: green.

## Step 9. Run the whole thing for the first time

```
uv run python src/run.py
```

On a laptop without a GPU this takes several minutes. On Colab with the GPU
turned on it is much faster.

Look at `results/run_curves.png` and `results/run_confusion.png`.

Check: it finished, it printed a train accuracy and a test accuracy, and the
loss curve goes down. If the two accuracies are identical, look at your split
again.

## Step 10. Run it three more times

```
uv run python src/run.py --scratch --tag scratch
uv run python src/run.py --freeze --lr 1e-3 --tag frozen
uv run python src/run.py <a setting of your own> --tag <a name for it>
```

Write each printed row into the table in your report as you go. Do not wait
until the end and try to remember.

Check: four rows in the table, and you can say how much starting from ImageNet
changed the test accuracy.

## Step 11. Write your cleaning rule

Not a coding step. Using what you saw in Step 3, write down in one or two
sentences what does not belong in each category. Do it before Step 12, so that
the suspect list does not decide the rule for you.

Check: someone else could apply your rule to your images and remove the same
ones you would.

## Step 12. Go through the suspects and the near copies

```
uv run python src/clean.py suspects
```

Open `results/cleaning/suspects/`. For each class sheet, go through the
suspects in order and decide each one by your rule. Then go through
`copies.png` and keep one of each pair. Remove with:

```
uv run python src/clean.py remove <class>/<file> <class>/<file> --reason "<which part of your rule>"
```

Then go back through the sheets from Step 3 for anything the suspect list did
not catch.

Check: `uv run python src/clean.py count` prints how many you removed from
each class and why, and you know how many of the first 20 suspects in each
class you removed.

## Step 13. Train on the clean data and compare fairly

```
uv run python src/run.py --tag clean
uv run python src/check.py --compare run clean
```

Check: you have the test accuracy before and after, and the accuracy on your
new images before and after, and you can say why the second pair is the fairer
comparison.

## Step 14. Write `worst_examples` and look at the mistakes

```
uv run pytest tests/test_evaluate.py
uv run python src/run.py --tag clean
```

Check: `results/clean_worst.png` shows ten pictures with what the model said
and what they really are.

## Step 15. Put the clean model on your page and check it

```
uv run python src/export_web.py --tag clean
uv run python src/check.py
```

Check: `check.py` says the exported model agrees with Python, and reports an
accuracy on your new images.

## Step 16. Open the demo on your own machine

```
uv run python -m http.server -d docs 8000
```

Open http://localhost:8000.

Check: the badge at the top is green. If it is red, stop and fix it. A red badge
means the page is not preparing images the way your `prepare_image` prepared
them for training, so everything it says is wrong. Read what the badge says,
fix `prepare_image`, and then train and export again.

## Step 17. Publish it

```
git add -A
git commit -m "trained model and demo"
git push
```

`docs/model.onnx` is about 45 MB, so this push takes a while. Do it once, with
the model you want to hand in, not after every experiment.

Then on the GitHub website: your repository, Settings, Pages, Source "Deploy
from a branch", Branch `main`, Folder `/docs`, Save.

Your agent can do the git part for you. If it goes in circles over `gh auth
login` for more than ten minutes, stop it and use the website. Getting the
command line tool authenticated is not what this assignment is about.

Check: you opened the address on your phone, away from your own wifi, and it
worked. If the page says it could not load the model, look at your repository
on the GitHub website and see whether `docs/model.onnx` and `docs/model.json`
are actually there.

## Step 18. Write the long report

Go back through the "Write this down" boxes in README.md in order, with your
agent. You should already have every number and every picture you need, from
the checks above.

Check: every number in it is one you saw printed by code you ran.

## Step 19. Write the short report

Close the agent. One page, by yourself, in Korean if that is your first
language. The four questions are in README.md.

This is the last step for a reason. You cannot answer question 2 or 4 until you
have seen your model fail, and you cannot answer question 3 unless you were
paying attention the whole way through.

Check: it fits on one page, and you wrote all of it.
