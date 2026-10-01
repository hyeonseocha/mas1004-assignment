# Assignment 1: Train an image classifier on your own images

MAS1004 Data & AI, 2026 Fall

Due: Friday 16 October, 22:00, on LMS.

## What you will hand in

A web page, on the internet, at an address you can send to anyone. It looks at
a picture and says which of your categories it thinks it is. The picture can
come from your webcam, from a file you upload, or from something you draw with
the mouse. The model runs inside the page itself, so it keeps working after you
close your laptop, and nobody's photos are sent anywhere.

The model is ResNet18, a network that was already trained on ImageNet, 1.2
million photographs of 1,000 kinds of thing. You replace its last layer so that
it answers with your categories, and train it further on images you collected,
of categories you chose.

You will also hand in two reports. One page that you write yourself, and a long
one that you write with your agent. More on both at the bottom.

## What you are given and what you write

You are given the parts where there is nothing to learn and a lot to get stuck
on: the image downloader, the cleaning tool, the export script, the web page,
the checker, and a test for each function you have to write.

You write these:

| file | functions | problem |
|---|---|---|
| `src/data.py` | `prepare_image`, `load_folder`, `split_train_test` | 2 |
| `src/train.py` | `build_model`, `train` | 3 |
| `src/evaluate.py` | `predict_logits`, `accuracy`, `confusion_matrix` | 3 |
| `src/evaluate.py` | `worst_examples` | 5 |

Each one has a docstring saying exactly what goes in and what comes out, and a
test that checks it. Read the docstring, ask your agent to write the body, run
the test, repeat until it is green.

`PLAN.md` is a plan for doing all of this, written out step by step. Give it to
your agent one step at a time. It is a real plan of the kind you should learn
to write yourself, so read it before you use it.

## Getting your own copy

This repository is a template, so you do not fork it and you do not work in it.

1. Make an account at https://github.com if you do not have one.
2. Open https://github.com/jdasam/mas1004-assignment1
3. Press "Use this template", then "Create a new repository".
4. Give it a name and keep it public. Public is what makes GitHub Pages free.
5. `git clone` your new repository onto your own machine.

Everything you do from now on happens in your copy, and you hand in its
address. Commit and push as you go. A commit you did not push has not been
handed in.

If you made your copy on 29 September before the starter code changed to
ResNet18 (your `src/data.py` has no `prepare_image` in it), make a new copy
from the template and move your `data/` folder into it. Your downloaded images
are not in git, so they do not come with the new copy on their own.

If you made your copy before 1 October, it has no `pyproject.toml`. Take the
uv setup from the template into your repository, in your repository folder:

```
git remote add template https://github.com/jdasam/mas1004-assignment1.git
git fetch template
git checkout template/main -- pyproject.toml uv.lock .python-version AGENTS.md CLAUDE.md .gitignore requirements.txt src/collect.py README.md PLAN.md
git commit -m "Use uv"
```

This replaces only the files named on the third line, none of the ones you
write.

## Setting up

On your own computer, use uv. It downloads the right version of Python and the
exact version of every package this assignment needs, and puts them in a
`.venv` folder inside your repository. You get the same versions on Windows,
macOS and Linux, and the Python you may already have is not touched.

Install uv once. On macOS, in Terminal:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, in PowerShell:

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then close the terminal, and quit and reopen VS Code, so that they can find
`uv`. `uv --version` should print a version number.

From now on, start every command with `uv run`. In your repository folder,
check that it is alive:

```
uv run pytest tests/test_export.py
```

The first time, this takes a few minutes, because uv downloads Python, PyTorch
and the rest. After that it starts at once. Do not `pip install` anything, and
do not change the versions in `pyproject.toml`: `uv.lock` holds exact versions
that were tested together. `AGENTS.md` tells your coding agent the same, and
most agents read it on their own.

Those tests cover code that was given to you, so they should pass before you
write a single line. If they do not, fix that first and ask for help if you
need it. Everything else will fail until you write it.

On Google Colab you do not need uv. The notebook installs everything with pip,
and there you leave `uv run` off every command.

Some computers cannot run this assignment, because PyTorch does not make
packages for them: Intel Macs, and Windows laptops with an ARM processor
(Snapdragon). uv stops with "not compatible with the lockfile's supported
environments" on them. Use Colab. On a Mac with Apple Silicon, you need macOS
14 (Sonoma) or later.

Training ResNet18 is much faster on a GPU. On a laptop without one, a run of
10 epochs on 400 images takes several minutes. Google Colab gives you a GPU
for free, and `assignment1_colab.ipynb` sets everything up there. Open it
directly in Colab with this link:
https://colab.research.google.com/github/jdasam/mas1004-assignment1/blob/main/assignment1_colab.ipynb

---

## Problem 1. Choose your categories and collect the images

Choose 3 to 5 categories. They have to be things you care about for some
reason, and you have to say what that reason is.

```
uv run python src/collect.py --classes "crossbody bag,pants,tshirt" --n 150
```

This downloads images into `data/raw/`. Then copy `data/raw` to `data/clean`
and work on the copy, so you always have the original to go back to.

Two things to think about when you choose. In Problem 5 you test the model on
images from a source that cannot overlap with your downloads, such as your own
photos, so choose things you can find such images of.
And the model can only use what is visible in the picture: an expensive wine
glass and a cheap one that look the same cannot be told apart by any model.
Because you start from a network that has already seen 1.2 million photos,
categories that differ in small details, such as similar breeds of dog, are
worth trying. Choose something you would still find interesting when the model
gets it wrong.

Write this down:
- Why these categories? Why do you care?
실제 사진을 구하기가 쉬울 것 같았습니다
- What do you think the model will actually use to tell them apart?
Pants (바지): 세로 방향으로 길게 늘어선 두 줄의 평행한 윤곽선과 하단부까지 이어지는 픽셀 분포 특성을 사용합니다.

Shirt (상의류): 어깨 선의 넓이, 소매 길이, 넥라인(V넥/라운드넥) 깊이, 중앙 버튼/지퍼 선의 경계 대비 차이로 구별합니다.

Crossbody bag (가방류): 직사각형 형태의 블록 모양, 중앙부의 짙은 픽셀 밀도, 그리고 어깨끈을 형성하는 가늘고 긴 얇은 선 패턴을 기준으로 판단합니다.
- How many images did you get for each? Show the counts.
카테고리	이미지 수
data/clean/crossbody_bag	150개
data/clean/pants	189개
data/clean/tshirt	158개

## Problem 2. Turn your folder into numbers

Write `prepare_image`, `load_folder` and `split_train_test` in `src/data.py`.

```
uv run pytest tests/test_data.py
```

`prepare_image` prepares one photo exactly the way every ImageNet photo was
prepared when ResNet18 was trained: the shorter side resized to 256, the
224 by 224 square in the middle cut out, and each colour channel scaled with
the ImageNet mean and standard deviation. The network learned to read photos
prepared like that, so it can only read yours if they are prepared the same
way. The web page does the same steps in JavaScript.

Your images are all different sizes and shapes, some are broken, and one of
them is a text file that ended up with a .jpg name. That is normal. The tests
put you through the same things.

Pay attention to the test called
`test_split_has_no_image_on_both_sides`. If the same picture is in the
training set and the test set, your test accuracy is not a measurement, it is a
memory. This is the single most common mistake in this assignment and it always
makes your numbers look better than they are.

Write this down:
- One of your images next to what `prepare_image` made of it. What was cut
  off? Find an image where the crop cut off part of the thing you care about.
![Before/After Comparison](before_after_comparison.png)


## Problem 3. Train it and report the first result

Write `build_model` and `train` in `src/train.py`, and `predict_logits`,
`accuracy` and `confusion_matrix` in `src/evaluate.py`.

```
uv run pytest tests/test_train.py tests/test_evaluate.py
uv run python src/run.py
```

`src/run.py` trains, measures, draws three pictures into `results/`, and saves
the model there. Run it at least these three ways, plus at least one more
setting of your own choosing, and fill in this table. Every run prints its row
for you. Do all of this before you clean anything: the run tagged `run` is the
"before" that Problem 4 compares against.

```
uv run python src/run.py
uv run python src/run.py --scratch --tag scratch
uv run python src/run.py --freeze --lr 1e-3 --tag frozen
```

| tag | start | trained | epochs | lr | trainable parameters | train accuracy | test accuracy |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

`--scratch` starts from random numbers instead of the ImageNet weights, as if
ImageNet had never happened. `--freeze` keeps every ImageNet weight as it is
and trains only the new last layer. With so few parameters to change it needs
larger steps, hence `--lr 1e-3`. Other useful things to change: `--epochs`,
`--lr`, `--batch-size`.

Write this down:
- The loss curve from `results/run_curves.png`. Did the loss go down? Did it
  keep going down, or did it flatten out?
- Your train accuracy is higher than your test accuracy. By how much? What does
  that gap mean?
- The confusion matrix from `results/run_confusion.png`. Which two categories
  does it mix up most? Does that surprise you?
- The same images, the same code, and the same number of epochs, starting from
  ImageNet and starting from random numbers. How far apart are the two test
  accuracies? Why do you think that is?
- `--freeze` trains about 0.01% of the parameters. How close does it get to
  training all of them?

## Problem 4. Clean your data and train again

This is the most important problem in the assignment.

Every image in `data/clean` got its label from a search engine. You typed
"wine glass", and whatever came back is now called `wine_glass`: drawings,
product catalogues, charts about types of wine glass, photos of a bar, the same
picture four times with different watermarks. The model learns every one of
them as a true example of its class. Your test set is cut from the same folder,
so it contains them too, and part of your test accuracy measures how well the
model agrees with the search engine.

Nobody can clean your data for you, because only you know what you meant by
each category. That is why you write the rule first.

`src/clean.py` is given to you for this problem. It works without any of the
code you wrote.

### 1. Look at every image

```
uv run python src/clean.py look
```

This draws every image in `data/clean` onto sheets in
`results/cleaning/look/`, with its file name under it. Open every sheet and
look at every picture. Note down the kinds of thing that should not be there.

### 2. Write your rule

Write down your rule before you remove anything, and follow it. "I removed
pictures where the object was not the main thing in the frame" is a rule.
"I removed the ones that looked wrong" is not.

### 3. Ask the rest of your data

```
uv run python src/clean.py suspects
```

For every image, this trains a small classifier on all the other images and
asks it what this one is. An image whose own label gets a low probability looks
unlike the rest of its class. `results/cleaning/suspects/<class>.png` shows
those images for each class, the most suspicious first.

`results/cleaning/suspects/copies.png` shows pairs of near copies: the same
picture resized, cropped a little, or saved again with a different watermark.
Downloaded images are full of them. If one copy ends up in your training set and
the other in your test set, the model passes the test by remembering, and your
test accuracy is higher than it should be. Keep one of each pair.

A suspect is not automatically junk. Some are good photos that are just
unusual, and those are exactly the ones your model needs most. Decide each one
by your rule.

### 4. Remove what breaks your rule

```
uv run python src/clean.py remove wine_glass/0063.jpg wine_glass/0069.jpg --reason "chart, not a photo"
```

This moves the files out of `data/clean` into `data/removed/` and records the
reason. Nothing is deleted, and nothing is changed in `data/raw`. Use this
rather than deleting files by hand, so that the counts in the next step are
right.

### 5. Count, train again, and compare fairly

```
uv run python src/clean.py count
uv run python src/run.py --tag clean
uv run python src/check.py --compare run clean
```

Cleaning changed your test set as well as your training set, because some of
the junk you removed was in the test set. So the test accuracy before and after
cleaning is measured on different images, and the two numbers cannot simply be
compared. Your new images from Problem 5 did not change, so `--compare`
measures both saved models on them. Collect them before you get to this step.

Write this down:
- Your rule, in one or two sentences, as you wrote it before you started.
- Three kinds of junk you found, with one picture of each.
- The table that `python src/clean.py count` prints.
- How many near copies it found, and what you did with them.
- Of the first 20 suspects in each class, how many did you remove? Did the
  suspects include junk you had missed when you looked yourself? Did you find
  junk that it did not list?
- The test accuracy before and after, and the accuracy on your new images
  before and after, from `--compare`. If a number went down, say so and think
  about why. That happens, and an honest explanation is worth more than a good
  number.

## Problem 5. Test it on images from a new source

Everything so far used images from one search. Your test set came from the
same download as your training set, so it shares its habits: the same kind of
product photo, the same white backgrounds, sometimes the very same picture
twice. Now test the model on images from somewhere else.

Collect at least 5 images per category from a source that you can be sure is
not in your downloaded images, and put them in `data/new_images/<category>/`,
using exactly the category names from your training folders. Collect them
early, because Problem 4 uses them too.

Save them as JPEG or PNG. An iPhone saves photos as HEIC (`.heic`) unless you
change it, and the code here cannot read HEIC. On the iPhone, Settings,
Camera, Formats, Most Compatible makes it save JPEG from then on. Photos you
already took in HEIC have to be converted to JPEG, which your coding agent can
do for you. `check.py` tells you if it finds a HEIC file.

What counts as such a source is up to you, as long as you can say why none of
its images can be among your downloads. Some that work:
- photographs you take yourself, which is the simplest
- photographs a friend took and sent you
- frames from a video you recorded
- photos from your own phone's gallery

Another search engine, or another search phrase, does not work. The same
product photos are copied onto every shopping site, so a second search brings
back many of the pictures you already have. Check with:

```
uv run python src/clean.py overlap
```

It compares every new image with every image you downloaded and draws the near
copies into `results/cleaning/overlap.png`. Any it finds are not new: take
them out of `data/new_images`, and think about whether the rest of that source
is really separate.

Then write `worst_examples` in `src/evaluate.py`, put your cleaned model on
your page, and run the checker:

```
uv run python src/export_web.py --tag clean
uv run python src/check.py
```

`check.py` measures the model that is now on your page on your new images, and
prints a confusion matrix and the mistakes it was most confident about.

Your accuracy here will probably be worse than your test accuracy from
Problem 3. That is not a mistake you made. Explaining it is the assignment. If
it is not worse, the images it does get wrong are still the ones to explain.

Write this down:
- Where your new images came from, and why you are sure none of them are among
  your downloads. The output of `python src/clean.py overlap`.
- The accuracy on your downloaded test set and the accuracy on your new images,
  side by side.
- Five mistakes it was confident about, with the pictures. For each one, what
  in that image do you think pushed it the wrong way?
- What is different about your new images? Background, lighting, angle,
  distance, what else is in the frame?

## Problem 6. Put the demo on the web

`src/export_web.py` wrote everything the page needs into `docs/`. Look at it
first on your own machine:

```
uv run python -m http.server -d docs 8000
```

and open http://localhost:8000. Opening `index.html` by double clicking will
not work, and neither will the webcam, because browsers only allow cameras on
`https://` pages and on `localhost`.

Check the badge at the top of the page. If it is red, the page is preparing
images differently from the way your `prepare_image` prepared them for
training, and the demo is lying to you. Fix that before you publish.

Then commit and push, and turn on GitHub Pages: your repository, Settings,
Pages, then Source "Deploy from a branch", Branch `main`, Folder `/docs`, Save.
Your demo appears at `https://<your name>.github.io/<your repository>/` after a
minute or two. The first time you look it is often a 404. Wait and reload.

The folder is called `docs` for exactly this reason. GitHub Pages publishes a
folder with that name and no other configuration.

`docs/model.onnx` is about 45 MB. Every time you commit a new one, another
45 MB goes into the history of your repository, so export and commit only the
model you want to publish, not every experiment. Check that `docs/model.onnx`
really is in your repository on the GitHub website. If you only see
`index.html` and `app.js`, you trained a model but never committed it, and your
published page will not load.

Write this down:
- Your GitHub Pages address.
- One sentence on what happens when you show it something that is none of your
  categories.

---

## What to submit on LMS

1. The address of your web demo.
2. The address of your code repository.
3. The short report.
4. The long report.
5. `data/new_images/` as a zip. Not the downloaded images, only the ones from
   your new source.

## The short report

One page. Not one and a bit. One.

Write it yourself. No language model, at any stage, including for tidying it up
afterwards. If Korean is your first language, write it in Korean. This is the
one piece of work in this course where neither your English nor your polish
counts for anything, and what you actually think counts for everything.

Four questions:

1. What does your model tell apart, and why did you pick that?
2. The worst mistake it makes. Which image, what did it answer, and what do
   you think made it answer that?
3. One thing your coding agent got wrong. What did it claim, what was actually
   true, and what made you look?
4. If you started again tomorrow, what would you do differently?

On the third question: everybody's agent gets something wrong. It writes code
that runs and does the wrong thing, or it reports an accuracy it never
measured, or it quietly changes something you told it not to. "Nothing went
wrong" will be read as "I did not check".

## The long report

As long as you like. Write it with your agent. That is what it is for, and it
is the right tool for this job.

Be aware of who reads it. It will be read by a program, and by you while you
are writing the short report. So write it for those two readers: complete,
ordered, every number traceable. Do not write an introduction, and do not
explain what a neural network is.

It holds:

- Everything the "Write this down" boxes asked for, in the order they appear
- Your experiment table, at least four rows
- The pictures from `results/` for every run you refer to
- The complete output of `python src/check.py`
- Your cleaning rule, the output of `python src/clean.py count`, and the
  suspects and near copies you removed or kept
- Where your new images came from, and the output of
  `python src/clean.py overlap`
- The accuracy on your downloaded test set and the accuracy on your new images,
  next to each other

## How this is graded

There are no points attached to the problems. Three things are looked at:

- Whether the demo works, at the address you gave, on someone else's computer
- Whether the numbers in your long report are the ones `check.py` actually
  prints
- The short report

A model that scores 95% with no explanation is worth less than one that scores
55% whose owner can tell you exactly which pictures it fails on and why.

## Using AI coding agents

Use them for the code and for the long report. That is what this course is
about. You are responsible for what you submit: you have to be able to explain
what you were trying to do, how you got your result, and what it means, even if
you cannot explain every line.

Three rules follow from that. Never write a number that you did not see printed
by code you ran. When the agent says it fixed something, run the test yourself
before you believe it. And write the short report with your own hands, because
it is the one place where the answer has to be yours.

## When you are stuck

- Run `uv run python src/check.py`. It works out the answers for itself from
  the files on disk, so when it disagrees with your own code, one of the two
  is wrong.
- Read the error at the bottom of the traceback, not the top.
- If a test fails, paste the whole test output to your agent, not your summary
  of it.
- `CUDA out of memory` means the batch does not fit on the GPU. Run again with
  a smaller `--batch-size`, such as 16.
- Ask in class. Both Tuesday and Thursday have time for this.
