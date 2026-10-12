"""Generate configs/workloads/prompts.txt: 1200 English prompts, one per line.

  python3 configs/workloads/make_prompts.py

Deterministic (fixed seed), so every phase and every box uses identical prompts.

Design constraints:
- Moderate length: 110-170 words each (~140-220 Llama-3 tokens), so the benchmark
  can truncate every prompt to exactly ISL=128 tokens.
- Varied OPENINGS: subject and opener come first. vLLM's automatic prefix caching
  reuses KV for shared leading token blocks; templated prompts with a common
  prefix would get a free partial prefill in Phase 2 but not in HF (Phase 1),
  making the stacks incomparable.
- Ask for long, open-ended answers (decode-heavy), though the benchmark also
  forces the output length.
- No newlines inside a prompt (one prompt per line).
"""
from __future__ import annotations

import random
from pathlib import Path

SEED = 1234
N_PROMPTS = 1200
MIN_WORDS, MAX_WORDS = 110, 170
OUT = Path(__file__).resolve().parent / "prompts.txt"

SUBJECTS = [
    "the water cycle", "plate tectonics", "photosynthesis", "black holes", "the immune system",
    "antibiotic resistance", "vaccines", "the human microbiome", "sleep", "circadian rhythms",
    "climate feedback loops", "ocean acidification", "coral reefs", "urban heat islands", "desalination",
    "nuclear fission reactors", "solar panels", "wind turbines", "grid-scale batteries", "hydrogen fuel cells",
    "electric vehicles", "high-speed rail", "container shipping", "air traffic control", "suspension bridges",
    "skyscraper foundations", "earthquake-resistant buildings", "3D printing", "semiconductor fabrication",
    "GPU architecture", "CPU caches", "virtual memory", "operating system schedulers", "file systems",
    "database indexes", "distributed consensus", "public-key cryptography", "password hashing",
    "content delivery networks", "the TCP handshake", "DNS", "compilers", "garbage collection",
    "version control", "unit testing", "continuous integration", "microservices", "message queues",
    "recommendation systems", "search engines", "neural networks", "transformer language models",
    "reinforcement learning", "computer vision", "speech recognition", "machine translation",
    "the printing press", "the Silk Road", "the Roman aqueducts", "the Industrial Revolution",
    "the invention of the telephone", "the space race", "the Apollo guidance computer", "the history of the internet",
    "medieval castles", "ancient Egyptian pyramids", "the Great Wall of China", "Viking navigation",
    "the Renaissance", "the scientific method", "double-entry bookkeeping", "compound interest",
    "inflation", "central banks", "supply chains", "index funds", "insurance", "auctions",
    "game theory", "behavioral economics", "public libraries", "national parks", "urban planning",
    "public transit design", "water treatment plants", "sewage systems", "recycling", "composting",
    "beekeeping", "sourdough baking", "coffee roasting", "cheese making", "fermentation",
    "knife sharpening", "woodworking joinery", "bicycle maintenance", "home electrical wiring",
    "indoor plant care", "vegetable gardening", "bird migration", "octopus intelligence",
    "wolf pack behavior", "honeybee communication", "tree root networks", "tidal pools", "deserts",
    "glaciers", "volcanoes", "hurricanes", "lightning", "rainbows", "the northern lights",
    "the Moon's phases", "Mars exploration", "exoplanet detection", "telescopes", "GPS",
    "atomic clocks", "music theory", "jazz improvisation", "orchestral conducting", "film editing",
    "animation", "typography", "chess openings", "marathon training", "strength training",
    "learning a second language", "memory techniques", "public speaking", "negotiation",
    "project management", "remote work", "technical writing", "code review", "on-call rotations",
]

OPENERS = [
    "{S} is a topic I keep coming back to.",
    "{S} came up in a conversation with a friend yesterday.",
    "I am preparing a short course that touches on {s}.",
    "My team asked me to explain {s} at our next meeting.",
    "While reading about {s}, I realized how little I understand it.",
    "A student asked me a question about {s} that I could not answer well.",
    "Lately I have been curious about {s}.",
    "{S} shows up in the news more and more often.",
    "Imagine you are guiding a curious newcomer through {s}.",
    "There is a lot of confusing material online about {s}.",
    "For a podcast episode, I need a solid grounding in {s}.",
    "I volunteered to write a newsletter piece on {s}.",
    "Our book club drifted into a debate about {s}.",
    "Before visiting a museum exhibit on {s}, I want to prepare.",
    "Thinking about {s} from first principles sounds like a good exercise.",
    "Someone told me that {s} is simpler than it looks.",
    "As a hobbyist, I have only scratched the surface of {s}.",
    "In my notes I have a half-finished outline about {s}.",
    "A documentary I watched last week covered {s} too quickly.",
    "When people mention {s}, they often skip the interesting details.",
    "Picture a long train ride with nothing to do except learn about {s}.",
    "My manager wants a plain-language briefing on {s}.",
]

CONTEXT = [
    "Most explanations I find are either far too shallow or buried in jargon, and I want something in between that still respects the details.",
    "I understand the basic idea, but I struggle to connect it to how things actually work in practice.",
    "I would like to understand not just what happens, but why it happens that way and what would change if the assumptions were different.",
    "The people I will share this with are smart but not specialists, so clarity matters more than completeness.",
    "I keep hearing contradictory claims, and I want to separate well-established facts from popular myths.",
    "I learn best from concrete examples, small analogies, and a clear sense of the moving parts.",
    "I want to be able to explain it later in my own words without notes.",
    "It would help to see how the pieces fit together, from the big picture down to the smallest important detail.",
    "I am particularly interested in the trade-offs involved and the reasons experts disagree.",
    "I have about an hour to read carefully, so depth is welcome as long as the structure is easy to follow.",
    "I want to avoid oversimplifications that will need to be unlearned later.",
    "Some historical background would help me understand why things ended up the way they are today.",
]

FORMATS = [
    "a detailed explanation", "a structured walkthrough", "a long-form essay", "a step-by-step guide",
    "an in-depth overview", "a thorough tutorial", "a narrative explanation", "a comprehensive primer",
    "a careful analysis", "an annotated outline followed by full prose",
]

AUDIENCES = [
    "a curious high school student", "a busy professional", "a retired engineer", "a first-year university student",
    "a journalist on deadline", "a software developer", "a science teacher", "a small business owner",
    "a policy maker", "an artist with no technical background", "a nurse", "a graduate student in another field",
]

ASPECTS = [
    "how it works at a basic level", "the key components and their roles", "a short history of how it developed",
    "common misconceptions", "the most important trade-offs", "how it fails and what happens when it does",
    "real-world examples", "how experts measure or evaluate it", "how it has changed over the last few decades",
    "what beginners usually get wrong", "the open questions that remain", "how it connects to everyday life",
    "the role of scale", "the costs involved", "safety and risk considerations", "the physics or logic underneath",
    "a worked example with numbers", "an analogy that captures the core idea", "where the analogy breaks down",
    "how it compares to the main alternatives", "what the future might look like", "practical tips for getting started",
]

CONSTRAINTS = [
    "Use clear section headings.", "Define every technical term the first time it appears.",
    "Include at least two concrete examples.", "Avoid bullet-point lists; write in full paragraphs.",
    "End with a short summary of the key takeaways.", "Point out where the evidence is uncertain.",
    "Keep the tone friendly but precise.", "Include a simple numerical example where it helps.",
    "Mention one or two common mistakes and how to avoid them.", "Suggest what to read or explore next.",
    "Do not assume prior knowledge beyond high school math.", "Highlight the single most important idea early on.",
    "Use an analogy, then explain its limits.", "Write as if speaking to one person directly.",
]

CLOSINGS = [
    "Please be thorough; a long and complete answer is better than a short one.",
    "Take your time and cover each part fully before moving on.",
    "I would rather have too much detail than too little.",
    "Please make the answer substantial enough to read in one sitting.",
    "Feel free to go deep wherever the topic rewards it.",
    "Please write it out in full rather than summarizing.",
]


def make_prompt(rng: random.Random, subject: str, opener: str) -> str:
    opener = opener.format(S=subject[0].upper() + subject[1:], s=subject)
    context = " ".join(rng.sample(CONTEXT, 2))
    aspects = rng.sample(ASPECTS, 4)
    task = (f"Please write {rng.choice(FORMATS)} of {subject} for {rng.choice(AUDIENCES)}, "
            f"covering {aspects[0]}, {aspects[1]}, {aspects[2]}, and {aspects[3]}.")
    constraints = " ".join(rng.sample(CONSTRAINTS, 3))
    return " ".join([opener, context, task, constraints, rng.choice(CLOSINGS)])


def main() -> None:
    rng = random.Random(SEED)
    prompts: list[str] = []
    used_pairs: set[tuple[str, str]] = set()
    while len(prompts) < N_PROMPTS:
        subject = SUBJECTS[len(prompts) % len(SUBJECTS)]  # even coverage of subjects
        # Never reuse an (opener, subject) pair: that pair is the first ~16+ tokens,
        # i.e. a full prefix-cache block in vLLM.
        opener = rng.choice(OPENERS)
        if (opener, subject) in used_pairs:
            continue
        p = make_prompt(rng, subject, opener)
        if not (MIN_WORDS <= len(p.split()) <= MAX_WORDS):
            continue
        used_pairs.add((opener, subject))
        prompts.append(p)
    rng.shuffle(prompts)
    OUT.write_text("\n".join(prompts) + "\n")
    print(f"wrote {len(prompts)} prompts to {OUT}")


if __name__ == "__main__":
    main()
