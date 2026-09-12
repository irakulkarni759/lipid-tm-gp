# What this is, in plain language

No background needed. No maths.

### → [Try it in your browser](https://irakulkarni759.github.io/lipid-tm-gp/)

Pick two lipids, drag the slider, read the answer. Nothing to install.

## The problem

Cell membranes are made of fatty molecules called lipids. A membrane has a
temperature at which it changes from stiff to floppy, a bit like butter going
from fridge-hard to spreadable. That temperature is called **Tm**.

Tm matters because it lets you build a drug carrier that stays sealed at body
temperature and pops open when gently warmed. To do that you have to hit a
specific Tm, and Tm depends on which lipids you mix and in what proportions.

The trouble is that there are a huge number of possible mixtures, and finding
the Tm of each one means running a lab experiment that takes days.

## What we built

A program you give a lipid mixture to, which tells you the temperature it will
melt at, in about a second.

You type something like "70% DMPC, 30% DPPC" and it answers "about 35 degrees,
give or take 5." You can also run it backwards. Tell it you want something that
melts at 40 degrees and it tells you what to mix.

## How it works, without the jargon

**It learned from past experiments.** We collected around 400 published Tm
measurements, plus a new set our own lab measured. The program studied those and
looked for the pattern connecting mixture to temperature.

**It understands lipids by their shape, not their name.** This is the important
bit. We do not tell the program "this is DMPC." We tell it "this molecule has
two tails, each 14 carbon atoms long, and they are straight rather than kinked."
Because it thinks in those terms, it can make a sensible guess about a lipid
nobody has ever tested, in the same way you could guess that an unfamiliar wax
with long straight chains will probably melt at a high temperature.

**It knows that not all measurements are equal.** Different lab instruments give
different answers for the same sample. Ours reads about 9 degrees hotter than
the standard method. Rather than blending those into a mush, the program learns
each instrument's bias and corrects for it, so old published numbers and our new
numbers can sit side by side honestly.

**It says when it does not know.** Every answer comes with a range attached.
Where we have lots of similar data the range is narrow, and where we have little
it is wide. This is what separates it from a program that simply always produces
a number.

## Why you can trust it

We tested it the hard way. We hid an entire lipid mixture from it, including
every repeat measurement, and made it predict that mixture with no help at all.
Then we put it back and hid a different one. We did this hundreds of times.

On average its guess was about 4 degrees away from the truth. For some mixtures
it was within a tenth of a degree.

More importantly, when the program says "give or take 5 degrees," the true
answer really does fall within 5 degrees about 95 times out of 100. Its
confidence is honest. A program that is confidently wrong is worse than useless,
and this one admits when it is guessing.

## How to actually use it

Open a terminal in this folder. There are three things you can ask.

**1. What does this mixture melt at?**

```
python3 ask.py DMPC 70 DPPC 30
```

```
  DMPC 70%  /  DPPC 30%

    melts at about 28.8 C, give or take 4.8
    likely between 19 and 38 C
    the model is confident
```

**2. What should I mix to hit a target temperature?**

```
python3 ask.py --target 42 DPPC DSPC
```

```
To melt at 42 C, mix:

    DPPC  94%
    DSPC   6%

Predicted 42.0 C, give or take 4.7.
```

If the target is impossible for that pair it says so instead of inventing an
answer.

**3. Which lipids does it know?**

```
python3 ask.py --list          the common ones
python3 ask.py --find SM       search by name
```

You can mix two, three, or more, and the percentages do not have to be exact
because it rescales them.

It accepts 341 lipids, but it is not equally good at all of them. Nine have been
measured in real mixtures and those answers are solid: **DPPC, DMPC, DOPC, DSPC,
POPC, Cholesterol, DMPG, PSM, SSM**. Another thirty or so are familiar lipids
that have only ever been measured on their own, so a mixture containing one is a
genuine guess. The rest are obscure entries from a literature compilation.

You do not have to memorise which is which. Ask, and read how wide the range
comes back. A narrow range means it has seen things like this before. A wide one
means go and measure it.

Add `--plate` to any question to get the answer on our plate-reader scale
instead of the calorimetry scale that published values use.

## What it is good for

- Trying out mixtures on a laptop before committing lab time to them
- Finding out that something is impossible early. DMPC and DPPC cannot be mixed
  to reach above 41 degrees in any proportion, so if you need 42 you must use
  different lipids
- Choosing which experiment to run next. The program can point at the mixture it
  is least sure about, which is the one that will teach us the most

## What it cannot do

- It is shaky on mixtures containing cholesterol, because we have very little
  cholesterol data. It reports a wide range there rather than pretending
- Only 28 true mixtures exist in the whole collected dataset, and mixtures are
  exactly what we care about. More of those is the single biggest improvement
  available
- A mixture of two lipids does not really melt at one exact temperature, it melts
  across a range. We predict the midpoint of that range and should say so

## What is in this folder

| | |
|---|---|
| `data/lipids.csv` | A list of the lipids involved and their chemical details |
| `data/measurements.csv` | Every experiment we gathered, one per row, with its temperature |
| `figures/` | Two pictures. One shows the predicted curve, one shows the accuracy test |
| `src/` | The program itself, split into steps |
| `run_all.py` | Runs everything from scratch and rebuilds all of the above |
| `README.md` | The technical version of this document |
