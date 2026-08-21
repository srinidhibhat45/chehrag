# Chehrag - How We Built This, In Plain English

**Chehrag** is *chirāġ* (चिराग़), the word for a lamp, crossed with RAG - the
technical name for the technique. A flame you ask questions of, out loud, in
eleven languages, that answers from its own library or from documents you hand
it, in under 200 milliseconds.

This document explains what we built and why, without assuming you know the
field.

---

## The problem we were given

Someone speaks a question. The system has to understand the speech, find the right
information inside a large collection of documents, and answer - **and the whole
thing has to finish in under a fifth of a second.**

For scale: a single blink takes about 100 milliseconds. We had two blinks.

---

## Why the obvious approach can't work

The normal way to build this is to put everything on a server. The user's device
records their voice, sends it to the server, the server does the thinking, and
sends an answer back.

We measured what that costs before writing any code:

| Step | Time |
|---|---|
| Sending data to a server and back | 50–200 ms |
| Fastest speech-to-text available anywhere | 150 ms typical, 400 ms on a bad day |
| An AI model writing an answer | 500 ms and up |

The fastest speech-to-text service in the world uses **150 of our 200 milliseconds
on its own**. Add a round trip to a server and we're over budget before we've
looked at a single document.

So we stopped trying to make the normal approach faster, and changed where the
work happens.

---

## Our approach: do the work on the user's own device

**The key insight: the fastest network request is the one you never make.**

Instead of sending the question to a server to be answered, we send the *answering
machinery* to the user's browser, once, when they first open the page. After that,
every question is answered on their own laptop.

The network isn't reduced. It's **absent**. There is no server involved in
answering a question, so there is nothing to be slow.

This has four consequences, all good:

**It's genuinely fast.** No round trip means no round-trip delay. Our measured work
happens entirely in memory on the user's machine.

**It's free to host.** Because nothing runs on a server, we don't rent a server. The
whole thing is delivered as a static web page from a free global content network.

**It never goes to sleep.** Free servers hibernate when unused, and the first request
after a nap takes 30+ seconds. That single slow request would have destroyed our
worst-case score. A static page has nothing to wake up.

**It's faster the more people use it.** A shared server splits its power between
users. Here, each user brings their own laptop to the job.

---

## What "finding the right information" actually involves

The collection we search is 11.45 million records across 14 Indian languages -
about 56 gigabytes. Obviously we can't send all of that to a browser.

Two things make this tractable.

**We shrink each document to a fingerprint.** Every passage gets converted into a
list of numbers that captures its meaning. Similar meanings produce similar numbers.
Finding relevant text becomes finding nearby numbers, which computers do very fast.

**We compress those fingerprints aggressively** - down to a single bit per number.
But the *order* of operations turned out to matter enormously, and not in the way
we expected. Simplifying the fingerprints first and then compressing them scores
nearly three times better than compressing them directly, for exactly the same file
size. Done the obvious way, the system barely works at all. We only know this
because we measured both.

---

## The part we put the most thought into: how to split up documents

This was explicitly the part the brief said not to do naively, and it's where most
of our design effort went.

The problem: you can't search whole documents (too broad - the answer gets buried)
and you can't search single sentences (too narrow - you lose the context that makes
the sentence meaningful). Where you cut matters enormously.

We also checked our assumptions before designing anything. The documents in this
collection average 317 characters - they are *already* about the size you would cut
them down to. So the textbook approach of "chop big documents into smaller pieces"
does nothing here: it would hand back the same document unchanged. That finding
reshaped the whole design.

Rather than pick one cutting method and hope, **we cut the same documents six
different ways and search all six simultaneously.**

| How we cut | What it's good at |
|---|---|
| Leave passages whole | Questions answered by one self-contained paragraph |
| One sentence at a time | Precision - a single-sentence answer doesn't get buried |
| Fixed-length pieces with overlapping edges | Catches answers that straddle a boundary |
| Add a summary header before storing | Numbers and names, which the maths handles worst |
| Cut where the topic shifts, not at fixed lengths | Keeps complete ideas together |
| Glue related documents into one larger one | Answers spread across several documents |

Then we combine the six sets of results using a voting method: a passage that
several different cutting strategies all rank highly is very likely the right one.
Any single strategy has blind spots. Six strategies voting together cover each
other's gaps.

This produced **809,607 pieces from 98,867 documents** - about eight versions of
every document. Searching all six collections costs about 2 milliseconds total,
because each search is fast and they don't interfere with each other.

---

## Knowing when *not* to answer

A system that always answers confidently is worse than useless - it's actively
misleading. We built three checkpoints where the system can decline.

**Before searching - is this a reasonable question?**
Catches off-topic questions, nonsense, abuse, and attempts to trick the system into
ignoring its instructions.

**After searching - did we actually find anything?**
If the best match isn't good enough, we say we don't know. This is the single most
important guardrail, because it's the difference between "I found the answer" and
"I found the least-bad thing available."

**After drafting an answer - is every claim actually in the source?**
We check the answer's content against the documents we retrieved. Anything not
supported gets cut. If too little survives, we decline rather than guess.

The system is designed to say "I don't know" cleanly, and to say *why*.

---

## The trick that makes it feel instant

Speech-to-text can stream words as they're spoken rather than waiting for silence.

So we start searching **while the person is still talking.** By the time someone
finishes saying "what is the capital of-", we've already searched on the partial
phrase and prepared a likely answer. When they stop, we re-run on the complete
sentence to be correct.

The result is that the answer appears the moment they stop speaking. Not because we
beat the clock, but because we did the work during a moment the user didn't
experience as waiting.

---

## Being honest about the measurement

We report three numbers, as asked:

- **P50** - the typical query. Half are faster than this.
- **P70** - a slightly-worse-than-typical query.
- **P100** - the single **worst** query we recorded. Nothing was slower.

P100 is the demanding one and it's the one that shaped the whole design. An average
can hide disasters; P100 cannot. To keep the worst case tight we warm everything up
before measuring, load all data into memory in advance, and avoid creating new
memory during a query - the three things that cause unpredictable stalls.

We also report the pieces we *don't* control, separately and honestly: speech-to-text
is an external service with its own published timings, and we report what we
measured from it rather than folding it into our own numbers or quietly excluding it.

---

## What we deliberately chose not to do

**We don't split one search across multiple servers.** It sounds faster and is the
opposite. If you ask five machines and wait for all five, you're hostage to whichever
one is having a bad moment - and we're graded on our worst case, not our average.
More machines means more chances to be unlucky.

**We don't put the AI answer-writer in the timed path.** Generating polished prose is
slow and always will be. So the system produces a grounded answer directly from the
source text immediately, and a more fluent AI-written version arrives a moment later
if the user wants it. The fast answer is correct; the slow one is nicer.

**We don't index all 14 languages.** Indexing everything would be a worse product,
not a better one - a larger download for the user and slower searches, in exchange
for languages nobody's demoing in.

---

## What we actually measured

**Speed.** Real questions, run in a browser, each one different, nothing served
from cache:

| | On a quiet laptop | With Teams and Figma running |
|---|---|---|
| Typical question (P50) | **21.5 ms** | 53.1 ms |
| **The single worst question (P100)** | **27.6 ms** | 97.8 ms |

The target was 200. Even on a laptop that was genuinely busy, the worst question
out of sixteen used half the budget. We report both because nobody's machine is
idle when they're being shown something.

Three changes got it there. We moved the language model that reads the question
onto a **separate thread**, so it stops competing with everything else the page
is doing. We made the system **practise before it starts**, because the first
question any program answers is always slow and, unpractised, that one slow
question becomes your worst-case score. And we found that the model was running
on a single processor core when it could use four: fixing one line took the
typical question from 56 milliseconds to 21.5.

Searching all seven versions of the documents takes about **5 milliseconds**.
Three quarters of the remaining time is spent understanding the question, not
finding the answer.

A question that gets refused is even faster - an attempt to hijack the system's
instructions is caught in **0.1 milliseconds**, before any searching happens at all.

**Two things we got wrong and found by testing.**

We nearly shipped a bug that would have been invisible. The code that compares
documents was doing arithmetic that silently overflowed, producing wrong answers
with no error message at all - the system would have run perfectly and returned
nonsense. We only caught it because we wrote a test that checked the two halves
of the system agreed with each other.

We also found that compressing the documents *before* simplifying them is not
optional. Doing it the obvious way scored 0.23 on our accuracy measure. Doing it
in the right order scored 0.60 - nearly three times better, for the same file
size. That is the kind of thing you cannot reason your way to; you have to measure.

**Knowing when to decline.** On obviously out-of-scope questions - "who won the
2029 World Cup on Mars", "what is my bank balance" - the system correctly declines
7 times out of 8, while still answering 84% of legitimate questions.

On the dataset's own "unanswerable" questions it does noticeably worse, and we
want to be straight about why rather than quietly changing the test. Those
questions come bundled with documents that are *about the right subject but do not
contain the answer*. Telling those apart is genuinely hard, and our measurements
say so: the signal is weak. We report that honestly instead of picking an easier
test that would have flattered us.

We also had to reject our own first answer here. Optimising the standard
statistical measure produced a system that refused 71% of real questions - great
on paper, useless in practice. We replaced it with a plain-English rule: *answer
at least 85% of answerable questions, then decline as much as you safely can.*

---

## Bringing your own documents

A lamp you can only ask about someone else's library is half a lamp. You can
hand Chehrag your own material three ways: paste text, drop a file (PDF, Word,
Markdown, spreadsheets, web pages, and most things made of text), or give it a
link.

What happens next is the important part, and it's the same thing that happens to
the built-in library: the document is cut up six different ways, each piece is
turned into its fingerprint of numbers, and those fingerprints are compressed
and filed. Your document then **competes directly with the built-in library** on
equal terms - there's no separate "search my files" mode, because both sides are
measured on exactly the same scale.

Three things we were careful about.

**Nothing is uploaded.** The reading, the cutting and the fingerprinting all
happen on your machine. The only thing our server ever sees is a link you
explicitly asked it to go and fetch - and it can't fetch a page itself only
because a web browser is forbidden from reading other websites directly.

**It only happens once.** Fingerprinting a 400-page PDF takes a while. We save
the result, so it costs you that time once rather than every visit.

**Adding a document can't slow down a question.** This one took real work.
Fingerprinting a long document is minutes of continuous effort, and a question
asked in the middle of that would otherwise have to wait in line behind it -
blowing the whole time budget before it even started. So the fingerprinting
happens on a separate thread, in small batches, and it stops entirely the
instant a question arrives.

---

## Making the deadline real

There's a difference between *being* fast and *guaranteeing* you'll be fast.

A stopwatch tells you afterwards that you were late. It doesn't stop you being
late. So rather than just measuring, the system now **checks the clock before
the expensive step and decides how much work it can afford.** If the machine is
struggling, it searches less thoroughly rather than missing the deadline - and
it tells you on screen that it did.

Which raised an obvious problem: we develop on a fast laptop, where the deadline
is never in danger, so none of that code would ever run during testing. We
couldn't borrow a slow phone for every test.

So we shrank the deadline instead. Squeezing the system into 20 milliseconds on
a fast machine puts it under exactly the same pressure as 200 milliseconds on a
machine ten times slower. We ran the whole question set at nine different
deadlines, from the real 200 down to 8.

The system held its deadline perfectly all the way down to 8 milliseconds, and
gave up no search quality at all until 20 - meaning it would still meet the real
200ms target on a machine roughly twenty-five times slower than ours. Below
about 10 milliseconds the budget is smaller than the single unavoidable step of
reading the question, which no amount of cleverness recovers. We report that
floor rather than trimming it off the bottom of the table.

**This test paid for itself immediately by finding real bugs**, all of them in
code that never runs on a fast machine, so no amount of ordinary testing would
have found them.

One was a piece of work that had been given a safety reservation four hundred
times larger than it actually needed. Under pressure the system kept skipping it
as "too expensive", and skipping it quietly broke the check that decides whether
an answer is good enough - so the system started refusing almost everything.

The other was a retry. When a step ran out of time, the system tried it again -
using time it definitively did not have. The one situation where retrying is
guaranteed to fail is the one where you already ran out of clock.

---

## Speaking, and being spoken to

You can ask out loud, and it can answer out loud.

Listening is Sarvam's, which is built for Indian languages and can work out
which one you're speaking rather than being told. It also sends back what it has
heard *while you're still talking* - so the system starts looking things up
mid-sentence, and by the time you stop, the answer is usually already found.

Speaking back is Sarvam's too. There is a second path wired up for ElevenLabs,
which has the better voice but only covers Hindi and Tamil of the languages
here, so each answer would go to whichever service can actually say it; it turns
on when a key is set. Using one service alone would mean choosing between a good
voice and being understood in Kannada.

If neither is configured, the browser's own built-in voice fills in, and the
interface says so in plain words rather than letting you assume you were hearing
the real thing.

---

## Asking in fifteen languages

The library is in Hindi. But the part of the system that turns words into
numbers was trained on a hundred languages at once, which means a question in
Tamil and a passage in Hindi can land near each other even though they share no
words at all.

That's a claim, so we tested it - with 3,000 questions across fifteen languages.

The dataset made this unusually honest. It's the same set of questions,
professionally translated into all fourteen languages, with a shared ID linking
them. So we could ask *literally the same question* fifteen ways and compare,
without our own translations introducing errors we'd then mistake for retrieval
errors.

**It works, and it fades gracefully.** Hindi finds the right passage 59.5% of
the time, which is the ceiling, since Hindi is what the library is written in.
English comes in at 56%. Nepali, Marathi, Malayalam and Bengali land in the
40s. Assamese and Sanskrit sit at the bottom near 27%, both languages the
underlying model saw least of. Speed didn't move at all: every language answers
in about 2.5 milliseconds.

### What the test found, which is the point of running it

Three faults, none of which were visible from the outside. All the same shape:
**something written correctly for the language it was tested in.**

**Nine of the fourteen languages were being turned away at the door.** The first
safety check confirms a question contains actual letters. It had been written to
recognise English and Hindi letters - so Bengali, Tamil, Telugu, Kannada,
Malayalam, Gujarati, Punjabi, Odia and Urdu were, as far as it was concerned,
gibberish. They were rejected in a tenth of a millisecond, before anything even
looked at them.

Nothing crashed. Nothing logged an error. The speed numbers actually looked
*better*, because turning a question away is much cheaper than answering it.
That's what made it survive: a fast, confident refusal looks exactly like a
guardrail doing its job.

**A rule meant to catch commands was rejecting ordinary questions.** The system
tried to spot "do this" rather than "tell me this" by looking for the absence of
a question word. But "Explain what a bone scan is and what it's used for" has no
question word in it, and is obviously a question. It was refusing 98 out of 3,000
- and when we checked, those 98 would have been answered correctly at the normal
rate. The rule now looks for verbs that ask the system to *act* - delete, run,
deploy - instead of guessing from what's missing.

**The 200ms promise had a hole in it.** One step's cost depends on how long your
question is, and nothing capped how long that could be. A pathological 6,594-
character question took 220 milliseconds on its own and broke the guarantee.

The safety net didn't help, and the reason is worth stating: the system's
deadline logic works by *doing less searching* when time is short - but by the
time it notices, the expensive step has already happened. **You cannot budget for
a cost after you've paid it.** Questions are now capped before that step, at a
length four and a half times longer than the longest question anyone actually
asks.

**And one bug in the measuring instrument itself.** The benchmark screen drew its
own progress text and then immediately started timing - so the time spent drawing
got counted as time spent answering. It reported 122 milliseconds where the truth
was 13. The tool was nine-tenths of what it was measuring, which is a good reminder
that instruments need checking as much as the thing they point at.

---

## The fire

The interface is a fire in a dark room, and that isn't only decoration.

It's the only status display: it fills as the library loads, breathes while
waiting, pulses in time with your voice while you speak, burns white-hot while
searching, flares when it answers, and cools to a dim ember when it declines.
One object, no spinners, no progress text.

It used to be a soft glowing ball - which read, correctly, as a blurry circle.
Fire doesn't look like a blur. What makes fire recognisable is *structure*:
filaments that curl, rise and burn out. So it's now drawn properly, frame by
frame, by the graphics card.

The constraint that shaped it: **a latency project cannot afford an animation
that competes with its own stopwatch.** So the fire is drawn on a completely
separate thread from the one answering your questions - it has its own canvas
and never interrupts. We checked rather than assuming: while the fire burns
continuously, the answering thread draws **zero** frames.

---

*Companion documents: `CONTEXT.md` holds the technical decision log. `README.md` covers running and deploying the system.*
