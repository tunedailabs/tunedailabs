"""
TunedAI Labs — Raw Passage Reasoning Test (30 Questions)
=========================================================
Tests the fine-tuned model against 30 verbatim pre-AI passages.
Sources: Hume (1748), Snow (1855), Mill (1843), Nightingale (1860)
All passages retrieved from Project Gutenberg. No AI wrote these sentences.

Usage:
  python raw_passage_test.py                  # Full test, both models
  python raw_passage_test.py --base-only      # Just base Qwen (faster)
  python raw_passage_test.py --start 10       # Resume from question 10
"""

import argparse
import json
import os
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL   = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_REPO = "tunedailabs/causal-reasoning-qwen-7b"
OUTPUT_FILE  = "raw_passage_results.json"
MAX_TOKENS   = 450

SYSTEM = (
    "You are a careful reasoner. Answer questions about causation, association, "
    "intervention, and counterfactuals precisely and correctly."
)

QUESTIONS = [

    # ── HUME (8 questions) ────────────────────────────────────────────────────

    {
        "id": 1,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section IV §22, 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"All reasonings concerning matter of fact seem to be founded on the relation of Cause and '
            'Effect. By means of that relation alone we can go beyond the evidence of our memory and '
            'senses. If you were to ask a man, why he believes any matter of fact, which is absent; '
            'for instance, that his friend is in the country, or in France; he would give you a reason; '
            'and this reason would be some other fact; as a letter received from him, or the knowledge '
            'of his former resolutions and promises."'
        ),
        "question": (
            "According to Hume in this passage, what is the only relation that allows us to reason "
            "beyond what we currently see and remember? Give a concrete example from the passage and "
            "explain the structure of that reasoning."
        ),
        "correct_answer": (
            "Cause and effect is the only relation that lets us go beyond present memory and senses. "
            "Example: a man believes his friend is in France because he received a letter — the letter "
            "(present fact) is taken as an effect of the friend writing from France (absent fact). "
            "The structure is: present observable fact → inferred cause or effect → conclusion about "
            "absent reality."
        ),
        "score_keywords": [
            ["cause and effect", "causal", "cause-and-effect", "relation of cause"],
            ["beyond", "memory", "senses", "absent", "go beyond"],
            ["letter", "reason", "other fact", "evidence"],
        ],
    },
    {
        "id": 2,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section IV §23, 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"I shall venture to affirm, as a general proposition, which admits of no exception, that '
            'the knowledge of this relation is not, in any instance, attained by reasonings a priori; '
            'but arises entirely from experience, when we find that any particular objects are constantly '
            'conjoined with each other. Let an object be presented to a man of ever so strong natural '
            'reason and abilities; if that object be entirely new to him, he will not be able, by the '
            'most accurate examination of its sensible qualities, to discover any of its causes or effects."'
        ),
        "question": (
            "Hume claims causal knowledge cannot be attained 'a priori.' What does this mean, and "
            "what does he say is the actual source of our causal knowledge? What is his evidence for "
            "this claim — the thought experiment he describes?"
        ),
        "correct_answer": (
            "'A priori' means through reason alone, without experience. Hume says causal knowledge "
            "arises entirely from experience — specifically from observing that objects are constantly "
            "conjoined. His thought experiment: even the most intelligent person, faced with a "
            "completely new object, cannot reason out its causes or effects by examining it. They "
            "must experience the object in relation to other things first."
        ),
        "score_keywords": [
            ["a priori", "reason alone", "without experience", "prior to experience"],
            ["experience", "constantly conjoined", "conjunction", "observation"],
            ["new object", "thought experiment", "cannot discover", "sensible qualities"],
        ],
    },
    {
        "id": 3,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section V (constant conjunction), 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"after the constant conjunction of two objects—heat and flame, for instance, weight and '
            'solidity—we are determined by custom alone to expect the one from the appearance of the '
            'other. This hypothesis seems even the only one which explains the difficulty, why we draw, '
            'from a thousand instances, an inference which we are not able to draw from one instance, '
            'that is, in no respect, different from them. Reason is incapable of any such variation. '
            'The conclusions which it draws from considering one circle are the same which it would '
            'form upon surveying all the circles in the universe. But no man, having seen only one '
            'body move after being impelled by another, could infer that every other body will move '
            'after a like impulse. All inferences from experience, therefore, are effects of custom, '
            'not of reasoning."'
        ),
        "question": (
            "Hume distinguishes between the conclusions we can draw from one instance vs. a thousand "
            "instances of two events occurring together. What explains this difference according to "
            "Hume — and why does he say 'reason' cannot account for it?"
        ),
        "correct_answer": (
            "The difference is explained by custom (habit), not reason. After constant conjunction "
            "over many instances, we develop a habitual expectation. Reason cannot explain this "
            "because reason draws the same conclusion from one instance as from a thousand — for "
            "example, one circle yields the same geometric truth as a million circles. But causation "
            "is not like geometry: a single impact doesn't justify inferring all future impacts. "
            "Only repeated experience builds that expectation."
        ),
        "score_keywords": [
            ["custom", "habit", "habitual"],
            ["reason", "reasoning", "cannot", "incapable"],
            ["thousand instances", "one instance", "repetition", "constant conjunction"],
        ],
    },
    {
        "id": 4,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section VII Part II §58, 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"It appears that, in single instances of the operation of bodies, we never can, by our '
            'utmost scrutiny, discover any thing but one event following another, without being able '
            'to comprehend any force or power by which the cause operates, or any connexion between '
            'it and its supposed effect. The same difficulty occurs in contemplating the operations '
            'of mind on body—where we observe the motion of the latter to follow upon the volition '
            'of the former, but are not able to observe or conceive the tie which binds together '
            'the motion and volition, or the energy by which the mind produces this effect. All '
            'events seem entirely loose and separate. One event follows another; but we never can '
            'observe any tie between them. They seem conjoined, but never connected."'
        ),
        "question": (
            "This passage contains Hume's famous claim that events are 'loose and separate.' What "
            "exactly does he say we can and cannot observe? What is the significance of the "
            "distinction between 'conjoined' and 'connected'?"
        ),
        "correct_answer": (
            "We can observe one event following another — the sequence. We cannot observe the force, "
            "power, or tie that binds cause to effect. 'Conjoined' means events occur together "
            "regularly. 'Connected' implies a necessary link, a real causal mechanism. Hume's point: "
            "we only ever see the former and never the latter. There is no observable causal "
            "connection — only observable sequence."
        ),
        "score_keywords": [
            ["loose and separate", "following another", "sequence", "one event follows"],
            ["force", "power", "tie", "mechanism", "cannot observe", "connexion"],
            ["conjoined", "connected", "distinction", "difference between"],
        ],
    },
    {
        "id": 5,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section VII Part II §59 (billiard balls), 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"The first time a man saw the communication of motion by impulse, as by the shock of two '
            'billiard balls, he could not pronounce that the one event was connected: but only that it '
            'was conjoined with the other. After he has observed several instances of this nature, he '
            'then pronounces them to be connected. What alteration has happened to give rise to this '
            'new idea of connexion? Nothing but that he now feels these events to be connected in his '
            'imagination, and can readily foretell the existence of one from the appearance of the '
            'other. When we say, therefore, that one object is connected with another, we mean only '
            'that they have acquired a connexion in our thought, and give rise to this inference, by '
            'which they become proofs of each other\'s existence."'
        ),
        "question": (
            "After observing several billiard ball collisions, a man says they are 'connected.' "
            "According to Hume, what has actually changed between the first observation and the "
            "several observations — and what has NOT changed?"
        ),
        "correct_answer": (
            "What changed: only the man's mental state. He now has a habit/feeling of connexion in "
            "his imagination. He can readily foretell one event from the other. What has NOT changed: "
            "anything in the physical world. The 'connexion' exists only in thought, not in the "
            "objects themselves. Hume concludes that 'connected' really just means 'we expect one "
            "from the other' — a mental habit, not a physical bond."
        ),
        "score_keywords": [
            ["imagination", "thought", "mind", "feeling", "mental", "habit"],
            ["nothing", "physical world", "objects", "unchanged", "no change"],
            ["foretell", "expect", "inference", "predict"],
        ],
    },
    {
        "id": 6,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section VII Part II §59, 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"It appears, then, that this idea of a necessary connexion among events arises from a '
            'number of similar instances which occur of the constant conjunction of these events; nor '
            'can that idea ever be suggested by any one of these instances, surveyed in all possible '
            'lights and positions. But there is nothing in a number of instances, different from every '
            'single instance, which is supposed to be exactly similar; except only, that after a '
            'repetition of similar instances, the mind is carried by habit, upon the appearance of '
            'one event, to expect its usual attendant, and to believe that it will exist. This '
            'connexion, therefore, which we feel in the mind, this customary transition of the '
            'imagination from one object to its usual attendant, is the sentiment or impression from '
            'which we form the idea of power or necessary connexion. Nothing farther is in the case."'
        ),
        "question": (
            "Hume says 'necessary connexion' is an idea that arises from repetition — but he also "
            "says nothing in a thousand instances is different from a single instance. How does he "
            "resolve this apparent contradiction? What IS different after many repetitions?"
        ),
        "correct_answer": (
            "The resolution: nothing is different in the external events themselves — the physical "
            "world is the same whether you've seen it once or a thousand times. What changes is "
            "purely internal: after repetition, the mind forms a habit of transition — it is 'carried "
            "by habit' to expect the usual attendant event. The idea of necessary connexion comes "
            "from this internal feeling/habit, not from anything observed in the world."
        ),
        "score_keywords": [
            ["habit", "custom", "carried by habit", "mind is carried"],
            ["nothing different", "external", "physical", "same", "nothing in the instances"],
            ["internal", "feeling", "sentiment", "impression", "expect"],
        ],
    },
    {
        "id": 7,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section VII Part II §60 (counterfactual), 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"Similar objects are always conjoined with similar. Of this we have experience. Suitably '
            'to this experience, therefore, we may define a cause to be an object, followed by another, '
            'and where all the objects similar to the first are followed by objects similar to the '
            'second. Or in other words where, if the first object had not been, the second never '
            'had existed."'
        ),
        "question": (
            "Hume offers two definitions of a cause. State both. Are they equivalent? Give an example "
            "where one definition gives a clear answer and the other becomes difficult to apply."
        ),
        "correct_answer": (
            "First: regularity/constant conjunction — A is a cause of B if all objects similar to A "
            "are followed by objects similar to B. Second: counterfactual — if A had not existed, B "
            "would not have existed. They are not fully equivalent. Example: overdetermination (two "
            "people independently poison a victim). Person A's poison: counterfactual test fails "
            "(victim still dies from B's poison), but regularity test may pass. The definitions "
            "diverge in cases of redundant causation."
        ),
        "score_keywords": [
            ["regularity", "constant conjunction", "always conjoined", "first definition"],
            ["counterfactual", "had not been", "never had existed", "second definition"],
            ["equivalent", "not equivalent", "differ", "diverge", "overdetermination", "example"],
        ],
    },
    {
        "id": 8,
        "source": "David Hume, An Enquiry Concerning Human Understanding, Section VII Part I, 1748",
        "retrieved_from": "Project Gutenberg #9662",
        "passage": (
            '"experience only teaches us, how one event constantly follows another; without instructing '
            'us in the secret connexion, which binds them together, and renders them inseparable. '
            'Thirdly, We learn from anatomy, that the immediate object of power in voluntary motion, '
            'is not the member itself which is moved, but certain muscles, and nerves, and animal '
            'spirits, and, perhaps, something still more minute and more unknown, through which the '
            'motion is successively propagated, ere it reach the member itself whose motion is the '
            'immediate object of volition. Can there be a more certain proof, that the power, by '
            'which this whole operation is performed, so far from being directly and fully known by '
            'an inward sentiment or consciousness, is, to the last degree mysterious and unintelligible?"'
        ),
        "question": (
            "Hume uses the example of voluntary motion (moving a limb) to make a point about causal "
            "power. What is his argument? What does anatomy reveal that supports his conclusion?"
        ),
        "correct_answer": (
            "Hume argues that even in the case we feel most directly — willing our own limbs to move "
            "— the causal power is not directly known. Anatomy reveals that the 'immediate object' "
            "of voluntary motion is not the limb itself but intermediate mechanisms: muscles, nerves, "
            "animal spirits, and unknown finer processes. The causal chain is always mediated through "
            "hidden intermediate steps. We cannot directly perceive the power even in our own will."
        ),
        "score_keywords": [
            ["voluntary motion", "limb", "will", "volition"],
            ["muscles", "nerves", "intermediate", "mediated", "propagated"],
            ["mysterious", "unknown", "hidden", "not directly", "cannot perceive"],
        ],
    },

    # ── SNOW (8 questions) ────────────────────────────────────────────────────

    {
        "id": 9,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd ed., 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"As cholera commences with an affection of the alimentary canal, and as we have seen '
            'that the blood is not under the influence of any poison in the early stages of this '
            'disease, it follows that the morbid material producing cholera must be introduced into '
            'the alimentary canal—must, in fact, be swallowed accidentally, for persons would not '
            'take it intentionally; and the increase of the morbid material, or cholera poison, '
            'must take place in the interior of the stomach and bowels."'
        ),
        "question": (
            "Snow uses the anatomy of cholera's early stages to rule out a theory. What theory does "
            "he rule out, and what does he conclude must be true about the route of infection? "
            "What type of causal reasoning is he using here?"
        ),
        "correct_answer": (
            "Snow rules out the airborne/miasma theory: because the blood is not poisoned in the "
            "early stages, the disease cannot have entered through the lungs (which would reach the "
            "blood first). He concludes the morbid material must be swallowed — ingested through the "
            "alimentary canal. This is mechanistic/eliminative causal reasoning: he traces the known "
            "site of action backward to constrain the possible routes of entry."
        ),
        "score_keywords": [
            ["airborne", "miasma", "atmosphere", "lungs", "rules out", "eliminat"],
            ["swallowed", "ingested", "alimentary canal", "stomach", "mouth"],
            ["mechanistic", "site of action", "backward", "route", "eliminative"],
        ],
    },
    {
        "id": 10,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd ed., 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"Diseases which are communicated from person to person are caused by some material '
            'which passes from the sick to the healthy, and which has the property of increasing '
            'and multiplying in the systems of the persons it attacks. In syphilis, small-pox, and '
            'vaccinia, we have physical proof of the increase of the morbid material, and in other '
            'communicable diseases the evidence of this increase, derived from the fact of their '
            'extension, is equally conclusive."'
        ),
        "question": (
            "Snow makes a general causal claim about communicable diseases. What is the claim? "
            "What two types of evidence does he offer to support it, and why does he consider both "
            "conclusive?"
        ),
        "correct_answer": (
            "The claim: communicable diseases are caused by a specific material that passes person "
            "to person and multiplies within the host. Evidence type 1: physical/direct — in "
            "syphilis, smallpox, and vaccinia there is observable physical material that increases. "
            "Evidence type 2: indirect/epidemiological — in other diseases, the fact that they "
            "spread (extend) from person to person proves the material is multiplying, even without "
            "direct physical proof. Both are considered conclusive because multiplication is the "
            "only mechanism that explains spread."
        ),
        "score_keywords": [
            ["material", "morbid material", "passes from", "transmitted", "specific agent"],
            ["physical proof", "direct evidence", "syphilis", "smallpox", "vaccinia"],
            ["extension", "spread", "multiplying", "indirect", "epidemiological"],
        ],
    },
    {
        "id": 11,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd ed., 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"The instances in which minute quantities of the ejections and dejections of cholera '
            'patients must be swallowed are sufficiently numerous to account for the spread of the '
            'disease; and on examination it is found to spread most where the facilities for this '
            'mode of communication are greatest. Nothing has been found to favour the extension of '
            'cholera more than want of personal cleanliness, whether arising from habit or scarcity '
            'of water, although the circumstance till lately remained unexplained."'
        ),
        "question": (
            "Snow identifies a pattern — cholera spreads most where a certain condition is greatest. "
            "What is the condition, and what does Snow say was surprising about this pattern before "
            "his theory? How does his causal theory resolve what 'till lately remained unexplained'?"
        ),
        "correct_answer": (
            "The condition is lack of personal cleanliness / scarcity of water. Before Snow's theory, "
            "this correlation was observed but unexplained — it was a known pattern without a "
            "mechanism. His causal theory resolves it: unclean hands contaminated with cholera "
            "evacuations transfer the fecal-oral material to food and mouth. Cleanliness prevents "
            "this transfer. The pattern was explained by identifying the causal mechanism."
        ),
        "score_keywords": [
            ["cleanliness", "scarcity of water", "hygiene", "unclean"],
            ["unexplained", "till lately", "previously", "known pattern without"],
            ["mechanism", "fecal", "hands", "food", "transfer", "explains"],
        ],
    },
    {
        "id": 12,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd ed., 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"As soon as I became acquainted with the situation and extent of this irruption of cholera, '
            'I suspected some contamination of the water of the much-frequented street pump in Broad '
            'Street, near the end of Cambridge Street; but on examining the water, on the evening of '
            'the 3rd September, I found so little impurity in it of an organic nature, that I hesitated '
            'to come to a conclusion. Further inquiry, however, showed me that there was no other '
            'circumstance or agent common to the circumscribed locality in which this sudden increase '
            'of cholera occurred, and not extending beyond it, except the water of the above mentioned pump."'
        ),
        "question": (
            "Snow hesitated to conclude the pump was the source because the water appeared clean. "
            "What overrode his hesitation? What does this tell us about the relative weight of "
            "direct physical tests vs. epidemiological pattern evidence in causal reasoning?"
        ),
        "correct_answer": (
            "The epidemiological pattern — elimination of every other common factor — overrode the "
            "physical test. Snow concluded the pump was the cause because it was the only agent "
            "common to the affected area. This shows that negative physical evidence (low organic "
            "impurity) can be outweighed by positive pattern evidence (nothing else in common). "
            "Causal reasoning from patterns can be more probative than direct tests when the "
            "physical mechanism is not yet understood."
        ),
        "score_keywords": [
            ["elimination", "no other", "only agent", "pattern", "common factor"],
            ["physical test", "chemical test", "impurity", "clean water", "overrode"],
            ["epidemiological", "pattern", "probative", "more weight", "outweighed"],
        ],
    },
    {
        "id": 13,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd ed., Chapter IX, 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"As there is no difference whatever, either in the houses or the people receiving the '
            'supply of the two Water Companies, or in any of the physical conditions with which they '
            'are surrounded, it is obvious that no experiment could have been devised which would '
            'more thoroughly test the effect of water supply on the progress of cholera than this, '
            'which circumstances placed ready made before the observer. The experiment, too, was on '
            'the grandest scale. No fewer than three hundred thousand people of both sexes, of every '
            'age and occupation, and of every rank and station, from gentlefolks down to the very '
            'poor, were divided into two groups without their choice, and, in most cases, without '
            'their knowledge."'
        ),
        "question": (
            "Snow claims this natural situation eliminates confounding. List every feature of this "
            "situation that Snow identifies as making the two groups equivalent. What would be "
            "missing from this comparison if people had chosen their own water supply?"
        ),
        "correct_answer": (
            "Features making groups equivalent: same houses, same people, same physical conditions, "
            "all ages, all sexes, all occupations, all ranks and stations. Division was without "
            "choice and without knowledge. If people had chosen their water supply, wealthier people "
            "might have chosen safer water, introducing socioeconomic confounding — the groups "
            "would differ in wealth, health baseline, and many other factors, not just water source."
        ),
        "score_keywords": [
            ["same houses", "same people", "no difference", "physical conditions", "equivalent"],
            ["without choice", "without knowledge", "did not choose", "self-selection"],
            ["confound", "wealth", "socioeconomic", "would differ", "bias"],
        ],
    },
    {
        "id": 14,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd ed., 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"consequently, as 286 fatal attacks of cholera took place, in the first four weeks of '
            'the epidemic, in houses supplied by the former Company, and only 14 in houses supplied '
            'by the latter, the proportion of fatal attacks to each 10,000 houses was as follows. '
            'Southwark and Vauxhall 71. Lambeth 5. The cholera was therefore fourteen times as fatal '
            'at this period, amongst persons having the impure water of the Southwark and Vauxhall '
            'Company, as amongst those having the purer water from Thames Ditton."'
        ),
        "question": (
            "Snow reports a 14-fold difference in mortality between the two water company districts. "
            "Does this quantitative difference alone prove that the water caused the higher mortality? "
            "What additional reasoning is required, and where did Snow provide it?"
        ),
        "correct_answer": (
            "The 14-fold difference alone does not prove causation — it could reflect confounding "
            "if the two groups differed in other ways. The additional reasoning required is the "
            "elimination of confounders — which Snow provided in the previous passage by showing "
            "the two groups were identical in all other respects (same houses, people, conditions, "
            "no knowledge/choice of water supply). The quantitative result is only causally "
            "interpretable because confounding has been ruled out."
        ),
        "score_keywords": [
            ["alone", "not prove", "does not prove", "insufficient", "correlation"],
            ["confound", "confounding", "other factors", "alternative explanation"],
            ["combination", "together", "with the previous", "ruled out", "eliminat"],
        ],
    },
    {
        "id": 15,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd ed., 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"It thus appears that the districts partially supplied with the improved water suffered '
            'much less than the others, although, in 1849, when the Lambeth Company obtained their '
            'supply opposite Hungerford Market, these same districts suffered quite as much as those '
            'supplied entirely by the Southwark and Vauxhall Company."'
        ),
        "question": (
            "Snow compares the same districts at two different points in time — 1849 and the current "
            "epidemic. What changed between the two periods, and why is this before-and-after "
            "comparison powerful evidence for his causal claim?"
        ),
        "correct_answer": (
            "What changed: the Lambeth Company's water source moved from Hungerford Market (near "
            "sewage) to Thames Ditton (cleaner). Before the change (1849), the districts suffered "
            "equally. After the change, districts with Lambeth water suffered much less. This is "
            "powerful because it uses the same population/geography — ruling out all neighbourhood "
            "characteristics as confounders. The only variable that changed was the water source, "
            "and mortality changed accordingly. It's a natural dose-response experiment over time."
        ),
        "score_keywords": [
            ["1849", "before", "same districts", "previously"],
            ["water source", "Lambeth", "changed", "moved", "improved water"],
            ["same population", "neighbourhood", "confound", "only variable", "natural experiment"],
        ],
    },
    {
        "id": 16,
        "source": "John Snow, On the Mode of Communication of Cholera, 2nd ed., 1855",
        "retrieved_from": "Project Gutenberg #72894",
        "passage": (
            '"I had an interview with the Board of Guardians of St. James\'s parish, on the evening '
            'of Thursday, 7th September, and represented the above circumstances to them. In '
            'consequence of what I said, the handle of the pump was removed on the following day."'
        ),
        "question": (
            "The pump handle removal is often cited as one of the first examples of public health "
            "intervention based on causal reasoning. What type of causal act is this — and what "
            "would a pure correlational study have recommended instead? Why does the distinction matter?"
        ),
        "correct_answer": (
            "This is an intervention — acting on the causal variable (pump access) rather than just "
            "observing. A pure correlational study would have recommended further study or general "
            "hygiene improvements without targeting the specific cause. The distinction matters "
            "because intervening on a cause produces a predictable, reproducible effect; intervening "
            "on a correlate may do nothing. Snow's removal of the handle operationalized his causal "
            "claim — it was a test of the causal model, not just a policy response."
        ),
        "score_keywords": [
            ["intervention", "intervening", "act on", "causal variable", "remove the cause"],
            ["correlational", "observational", "would not", "further study"],
            ["test", "operationalize", "reproducible", "causal model", "distinction"],
        ],
    },

    # ── MILL (8 questions) ────────────────────────────────────────────────────

    {
        "id": 17,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, First Canon, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"If two or more instances of the phenomenon under investigation have only one circumstance '
            'in common, the circumstance in which alone all the instances agree, is the cause '
            '(or effect) of the given phenomenon."'
        ),
        "question": (
            "State Mill's First Canon (Method of Agreement) in plain language. Give an example of "
            "how you would apply it to determine the cause of food poisoning at a restaurant, and "
            "identify its main limitation."
        ),
        "correct_answer": (
            "Method of Agreement: if multiple cases of the phenomenon share only one circumstance, "
            "that circumstance is the cause. Food poisoning example: find all sick diners, identify "
            "the only dish they all ate — that dish is the likely cause. Main limitation: it cannot "
            "establish causation if the instances share multiple circumstances (confounding). It also "
            "cannot rule out that the shared circumstance is an effect rather than a cause."
        ),
        "score_keywords": [
            ["only one circumstance", "single common", "shared", "all instances agree"],
            ["food", "example", "sick", "applied", "apply"],
            ["limitation", "confounding", "multiple", "cannot rule out", "weakness"],
        ],
    },
    {
        "id": 18,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, Second Canon, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"If an instance in which the phenomenon under investigation occurs, and an instance in '
            'which it does not occur, have every circumstance in common save one, that one occurring '
            'only in the former; the circumstance in which alone the two instances differ, is the '
            'effect, or the cause, or an indispensable part of the cause, of the phenomenon."'
        ),
        "question": (
            "State Mill's Second Canon (Method of Difference) in plain language. Why is this method "
            "considered stronger than the Method of Agreement for establishing causation?"
        ),
        "correct_answer": (
            "Method of Difference: compare one instance where the phenomenon occurs with one where "
            "it doesn't, and they differ in only one circumstance — that circumstance is the cause. "
            "It is stronger than Agreement because it directly controls for all other variables. "
            "Agreement finds what's common to positive cases but can't rule out other common factors. "
            "Difference eliminates all alternative explanations by holding everything constant except "
            "the one variable — making it the logic behind controlled experiments."
        ),
        "score_keywords": [
            ["every circumstance", "save one", "only one difference", "controlled"],
            ["stronger", "stronger than", "better than", "more powerful"],
            ["controls", "eliminates", "rules out", "alternative", "all other variables"],
        ],
    },
    {
        "id": 19,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, Second Canon example, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"It is scarcely necessary to give examples of a logical process to which we owe almost '
            'all the inductive conclusions we draw in daily life. When a man is shot through the '
            'heart, it is by this method we know that it was the gunshot which killed him: for he '
            'was in the fullness of life immediately before, all circumstances being the same, '
            'except the wound."'
        ),
        "question": (
            "In the gunshot example, identify the two instances Mill is comparing, what they share, "
            "and what is different. Then explain why Mill says this method underlies 'almost all "
            "the inductive conclusions we draw in daily life.'"
        ),
        "correct_answer": (
            "Two instances: the man immediately before being shot (alive, all circumstances the same) "
            "and the man after being shot (dead, same circumstances except for the wound). Shared: "
            "everything else. Different: the gunshot wound. Mill says this method underlies daily "
            "life inference because we constantly reason by comparing a before-state to an after-state "
            "where one thing changed — this is the basic structure of everyday causal judgment."
        ),
        "score_keywords": [
            ["before", "after", "two instances", "fullness of life", "immediately before"],
            ["all circumstances", "same except", "one thing changed", "wound"],
            ["daily life", "everyday", "constantly", "basic structure", "before and after"],
        ],
    },
    {
        "id": 20,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"Of these methods, that of Difference is more particularly a method of artificial '
            'experiment; while that of Agreement is more especially the resource employed where '
            'experimentation is impossible. A few reflections will prove the fact, and point out '
            'the reason of it. It is inherent in the peculiar character of the Method of Difference, '
            'that the nature of the combinations which it requires is much more strictly defined '
            'than in the Method of Agreement. The two instances which are to be compared with one '
            'another must be exactly similar, in all circumstances except the one which we are '
            'attempting to investigate."'
        ),
        "question": (
            "Mill distinguishes between when the Method of Agreement vs. the Method of Difference "
            "is used. In what situations does each apply, and what practical constraint makes the "
            "Method of Difference harder to use in the real world?"
        ),
        "correct_answer": (
            "Method of Difference is the method of artificial experiment — used when you can control "
            "conditions and manipulate variables. Method of Agreement is used when experimentation "
            "is impossible and you must rely on naturally occurring cases. The practical constraint "
            "on Method of Difference: the two instances must be exactly similar in all circumstances "
            "except the one being investigated — this is hard to achieve in the real world where you "
            "cannot fully control all variables."
        ),
        "score_keywords": [
            ["experiment", "artificial", "controlled", "manipulate", "laboratory"],
            ["impossible", "cannot experiment", "observational", "naturally occurring"],
            ["exactly similar", "must be the same", "all circumstances", "constraint", "difficult"],
        ],
    },
    {
        "id": 21,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, Method of Agreement example, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"For example, let the antecedent A be the contact of an alkaline substance and an oil. '
            'This combination being tried under several varieties of circumstances, resembling each '
            'other in nothing else, the results agree in the production of a greasy and detersive '
            'or saponaceous substance: it is therefore concluded that the combination of an oil and '
            'an alkali causes the production of a soap. It is thus we inquire, by the Method of '
            'Agreement, into the effect of a given cause."'
        ),
        "question": (
            "Mill's soap example demonstrates the Method of Agreement in a scientific context. "
            "What is the antecedent (cause), what is the phenomenon (effect), and how does "
            "varying the other circumstances strengthen the causal conclusion?"
        ),
        "correct_answer": (
            "Antecedent/cause: contact of an alkali and an oil. Phenomenon/effect: production of "
            "soap (greasy, detersive, saponaceous substance). By trying the combination 'under "
            "several varieties of circumstances, resembling each other in nothing else,' Mill shows "
            "that the only common factor across all cases is the alkali + oil combination. This "
            "variation strengthens the conclusion by ruling out other potential causes — if soap "
            "forms regardless of what else varies, the alkali-oil combination must be the cause."
        ),
        "score_keywords": [
            ["alkali", "oil", "antecedent", "cause", "contact"],
            ["soap", "saponaceous", "effect", "phenomenon", "result"],
            ["varying", "several varieties", "nothing else", "only common factor", "rules out"],
        ],
    },
    {
        "id": 22,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, Sixth Canon, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"Whatever phenomenon varies in any manner whenever another phenomenon varies in some '
            'particular manner, is either a cause or an effect of that phenomenon, or is connected '
            'with it through some fact of causation."'
        ),
        "question": (
            "State Mill's Method of Concomitant Variations in plain language. Why does Mill include "
            "the phrase 'or is connected with it through some fact of causation' — what possibility "
            "does that cover?"
        ),
        "correct_answer": (
            "Method of Concomitant Variations: if two phenomena vary together (when one changes, "
            "the other changes), they are causally related — either one causes the other, or both "
            "are effects of a common cause. The phrase 'connected through some fact of causation' "
            "covers the common cause scenario: A and B may both be caused by C, making them "
            "correlated without either causing the other. Mill is acknowledging that covariation "
            "does not prove direct causation — it could indicate confounding by a third variable."
        ),
        "score_keywords": [
            ["varies together", "covariation", "correlation", "when one changes"],
            ["common cause", "third variable", "both effects", "confound"],
            ["does not prove", "direct causation", "could be", "alternative"],
        ],
    },
    {
        "id": 23,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, Concomitant Variations example, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"In the case of heat, for example, by increasing the temperature of a body we increase '
            'its bulk, but by increasing its bulk we do not increase its temperature; on the contrary '
            '(as in the rarefaction of air under the receiver of an air-pump), we generally diminish '
            'it: therefore heat is not an effect, but a cause, of increase of bulk."'
        ),
        "question": (
            "Mill uses the heat/bulk example to determine the direction of causation. What logical "
            "test does he apply? How does this test relate to the modern concept of intervention?"
        ),
        "correct_answer": (
            "Mill applies the asymmetric intervention test: he increases temperature and observes "
            "bulk increases; then he increases bulk (by rarefaction) and observes temperature "
            "decreases (or doesn't increase). The asymmetry — that manipulation flows only one way "
            "productively — reveals the causal direction. This directly anticipates modern do-calculus: "
            "do(heat) → bulk changes, but do(bulk) ↛ heat. Intervention on the cause affects the "
            "effect; intervention on the effect does not affect the cause."
        ),
        "score_keywords": [
            ["intervention", "manipulate", "increasing", "asymmetry", "direction"],
            ["do(", "do-calculus", "intervene", "one way", "asymmetric"],
            ["cause not effect", "heat is the cause", "direction of causation"],
        ],
    },
    {
        "id": 24,
        "source": "John Stuart Mill, A System of Logic, Book III Chapter 8, Method of Residues, 1843",
        "retrieved_from": "Project Gutenberg #27942",
        "passage": (
            '"Subducting from any given phenomenon all the portions which, by virtue of preceding '
            'inductions, can be assigned to known causes, the remainder will be the effect of the '
            'antecedents which had been overlooked, or of which the effect was as yet an unknown '
            'quantity. Suppose, as before, that we have the antecedents A B C, followed by the '
            'consequents a b c, and that by previous inductions... we have ascertained the causes '
            'of some of these effects... and are thence apprised that the effect of A is a, and '
            'that the effect of B is b. Subtracting the sum of these effects from the total '
            'phenomenon, there remains c, which now, without any fresh experiments, we may know '
            'to be the effect of C."'
        ),
        "question": (
            "Describe Mill's Method of Residues in plain language. How is this method used in "
            "scientific discovery — give a real-world example of where this type of reasoning "
            "has been applied."
        ),
        "correct_answer": (
            "Method of Residues: if you know what causes account for parts of a complex phenomenon, "
            "the unexplained remainder must be caused by remaining antecedents. In plain language: "
            "subtract known causes/effects until the residue points to an unknown cause. Classic "
            "example: the discovery of Neptune — astronomers knew the gravitational effects of all "
            "known planets on Uranus's orbit, subtracted those, and the residual perturbation "
            "pointed to an unobserved planet. Adams and Le Verrier predicted Neptune's position "
            "purely from the residue."
        ),
        "score_keywords": [
            ["subtract", "remainder", "residue", "remaining", "what's left"],
            ["Neptune", "planet", "Uranus", "perturbation", "unknown cause", "discovery"],
            ["scientific discovery", "points to", "unobserved", "unknown quantity"],
        ],
    },

    # ── NIGHTINGALE (6 questions) ─────────────────────────────────────────────

    {
        "id": 25,
        "source": "Florence Nightingale, Notes on Nursing: What It Is, and What It Is Not, 1860",
        "retrieved_from": "Project Gutenberg #17366",
        "passage": (
            '"Shall we begin by taking it as a general principle--that all disease, at some period '
            'or other of its course, is more or less a reparative process, not necessarily '
            'accompanied with suffering: an effort of nature to remedy a process of poisoning or '
            'of decay, which has taken place weeks, months, sometimes years beforehand, unnoticed, '
            'the termination of the disease being then, while the antecedent process was going on, '
            'determined?"'
        ),
        "question": (
            "Nightingale proposes that disease is a 'reparative process' caused by an earlier, "
            "unnoticed antecedent process. What does this mean for how we identify the cause of "
            "a disease? What error in causal reasoning does she implicitly warn against?"
        ),
        "correct_answer": (
            "Nightingale means that the visible disease (symptoms, crisis) is not the cause but the "
            "effect of an earlier process of poisoning or decay that may have begun years before — "
            "and went unnoticed. Identifying the cause requires tracing back to that earlier process, "
            "not treating the symptoms as the starting point. She implicitly warns against confusing "
            "the proximate cause (the disease crisis) with the distal cause (the earlier decay/poisoning) "
            "— and against treating an effect (the reparative process) as if it were the cause."
        ),
        "score_keywords": [
            ["earlier", "antecedent", "unnoticed", "weeks months years", "prior"],
            ["proximate", "distal", "confusion", "not the cause", "effect not cause"],
            ["trace back", "origin", "reparative", "symptom", "remedy"],
        ],
    },
    {
        "id": 26,
        "source": "Florence Nightingale, Notes on Nursing: What It Is, and What It Is Not, 1860",
        "retrieved_from": "Project Gutenberg #17366",
        "passage": (
            '"In watching disease, both in private houses and in public hospitals, the thing which '
            'strikes the experienced observer most forcibly is this, that the symptoms or the '
            'sufferings generally considered to be inevitable and incident to the disease are very '
            'often not symptoms of the disease at all, but of something quite different--of the want '
            'of fresh air, or of light, or of warmth, or of quiet, or of cleanliness, or of '
            'punctuality and care in the administration of diet, of each or of all of these."'
        ),
        "question": (
            "Nightingale identifies a common causal attribution error in medical care. What is the "
            "error, and what does she say is the actual cause of many 'disease symptoms'? Why is "
            "this distinction clinically important?"
        ),
        "correct_answer": (
            "The error: attributing patient suffering to the disease itself when it is actually "
            "caused by environmental deficiencies — lack of fresh air, light, warmth, quiet, "
            "cleanliness, proper diet. The actual causes are nursing/environmental factors, not "
            "the disease. Clinically important because: if you misidentify the cause as 'disease,' "
            "you do nothing about the actual causes. If you correctly identify the environmental "
            "cause, you can intervene and reduce suffering without treating the disease itself."
        ),
        "score_keywords": [
            ["attribution error", "misidentify", "not the disease", "something quite different"],
            ["fresh air", "light", "warmth", "cleanliness", "diet", "environment"],
            ["intervene", "can fix", "clinically", "treatment", "reduce suffering"],
        ],
    },
    {
        "id": 27,
        "source": "Florence Nightingale, Notes on Nursing: What It Is, and What It Is Not, 1860",
        "retrieved_from": "Project Gutenberg #17366",
        "passage": (
            '"The causes of the enormous child mortality are perfectly well known; they are chiefly '
            'want of cleanliness, want of ventilation, want of white-washing; in one word, defective '
            'household hygiene. The remedies are just as well known; and among them is certainly '
            'not the establishment of a Child\'s Hospital."'
        ),
        "question": (
            "Nightingale distinguishes between the known cause of child mortality and an incorrect "
            "remedy. What is the cause, what is the incorrectly proposed remedy, and why does "
            "Nightingale reject it? What principle of causal reasoning does her argument illustrate?"
        ),
        "correct_answer": (
            "Cause: defective household hygiene — lack of cleanliness, ventilation, whitewashing. "
            "Incorrect remedy: a Children's Hospital. Nightingale rejects it because the hospital "
            "does not address the actual cause — it treats symptoms after the fact rather than "
            "preventing the environmental conditions that cause child deaths. The principle: "
            "interventions must target the actual cause to be effective. Treating effects (sick "
            "children) while leaving the cause (poor household hygiene) intact does not reduce mortality."
        ),
        "score_keywords": [
            ["hygiene", "cleanliness", "ventilation", "household", "cause"],
            ["hospital", "does not address", "does not target", "treats symptoms"],
            ["target the cause", "intervene on cause", "must address", "prevent"],
        ],
    },
    {
        "id": 28,
        "source": "Florence Nightingale, Notes on Nursing: What It Is, and What It Is Not, 1860",
        "retrieved_from": "Project Gutenberg #17366",
        "passage": (
            '"In laying down the principle that the first object of the nurse must be to keep the '
            'air breathed by her patient as pure as the air without, it must not be forgotten that '
            'everything in the room which can give off effluvia, besides the patient, evaporates '
            'itself into his air. And it follows that there ought to be nothing in the room, '
            'excepting him, which can give off effluvia or moisture."'
        ),
        "question": (
            "Nightingale derives an intervention policy from a causal principle. State the causal "
            "principle, show how the policy follows from it, and explain why this is an example "
            "of causal reasoning rather than simple rule-following."
        ),
        "correct_answer": (
            "Causal principle: everything in the room that can give off effluvia contaminates the "
            "patient's air. Intervention policy: remove all such objects from the room (except the "
            "patient). This follows causally: if object X causes air impurity, then do(remove X) "
            "removes that contribution to impurity. This is causal reasoning — not just following "
            "a rule — because the policy is derived from an understanding of the causal mechanism "
            "(effluvia → air contamination → harm). A different mechanism would yield a different policy."
        ),
        "score_keywords": [
            ["effluvia", "air", "contaminates", "impurity", "evaporate"],
            ["remove", "do(remove", "intervention", "policy", "follows from"],
            ["mechanism", "derived from", "causal", "not just a rule", "understanding"],
        ],
    },
    {
        "id": 29,
        "source": "Florence Nightingale, Notes on Nursing: What It Is, and What It Is Not, 1860",
        "retrieved_from": "Project Gutenberg #17366",
        "passage": (
            '"I have known a medical officer keep his ward windows hermetically closed, thus '
            'exposing the sick to all the dangers of an infected atmosphere, because he was afraid '
            'that, by admitting fresh air, the temperature of the ward would be too much lowered. '
            'This is a destructive fallacy. To attempt to keep a ward warm at the expense of making '
            'the sick repeatedly breathe their own hot, humid, putrescing atmosphere is a certain '
            'way to delay recovery or to destroy the sick."'
        ),
        "question": (
            "Nightingale describes a medical officer acting on an incorrect causal model. What is "
            "his causal model, what is the correct causal model, and what is the consequence of "
            "acting on the wrong model?"
        ),
        "correct_answer": (
            "Wrong causal model: cold air causes harm → close windows to maintain warmth → patients "
            "recover. The officer correctly identifies warmth as protective but wrongly identifies "
            "cold air (rather than foul air) as the hazard. Correct causal model: foul/putrescing "
            "air causes harm → fresh air prevents this harm → open windows despite cold. Consequence "
            "of the wrong model: patients repeatedly breathe contaminated air, delaying recovery "
            "or causing death. Acting on the wrong causal variable (temperature instead of air quality) "
            "makes the intervention harmful."
        ),
        "score_keywords": [
            ["wrong model", "incorrect", "cold air", "temperature", "confused"],
            ["foul air", "putrescing", "contaminated", "air quality", "correct cause"],
            ["consequence", "harmful", "delay recovery", "acting on wrong", "wrong variable"],
        ],
    },
    {
        "id": 30,
        "source": "Florence Nightingale, Notes on Nursing: What It Is, and What It Is Not, 1860",
        "retrieved_from": "Project Gutenberg #17366",
        "passage": (
            '"In comparing the deaths of one hospital with those of another, any statistics are '
            'justly considered absolutely valueless which do not give the ages, the sexes, and '
            'the diseases of all the cases. It does not seem necessary to mention this. It does '
            'not seem necessary to say that there can be no comparison between old men with '
            'dropsies and young women with consumptions. Yet the cleverest men and the cleverest '
            'women are often heard making such comparisons, ignoring entirely sex, age, disease, '
            'place--in fact, all the conditions essential to the question. It is the merest gossip."'
        ),
        "question": (
            "Nightingale calls unadjusted hospital mortality comparisons 'absolutely valueless.' "
            "Name the statistical problem she identifies, list the variables she says must be "
            "controlled for, and explain with her own example why ignoring them misleads."
        ),
        "correct_answer": (
            "Statistical problem: confounding — hospitals treat patients of different types, so "
            "raw mortality comparisons measure patient mix, not care quality. Variables to control: "
            "age, sex, disease type, place. Her example: comparing 'old men with dropsies' to "
            "'young women with consumptions' is invalid because these groups have completely "
            "different baseline mortality. A hospital treating mostly elderly dropsical patients "
            "will show higher mortality than one treating young consumptives — not because it "
            "provides worse care, but because of case mix. Without adjustment, the comparison "
            "is meaningless."
        ),
        "score_keywords": [
            ["confound", "case mix", "patient mix", "different types", "baseline mortality"],
            ["age", "sex", "disease", "adjust", "control for", "stratify"],
            ["old men", "dropsies", "young women", "consumptions", "example", "baseline different"],
        ],
    },
]


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_answer(answer: str, keyword_groups: list) -> tuple:
    answer_lower = answer.lower()
    hits = sum(
        1 for group in keyword_groups
        if any(kw.lower() in answer_lower for kw in group)
    )
    return hits, len(keyword_groups)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("Loading base model (~14GB, may take 2-3 min)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Loading TunedAI Labs adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_REPO)
    model.eval()
    print("✓ Ready.\n")
    return tokenizer, model


def ask(question: str, tokenizer, model, use_adapter: bool) -> str:
    if use_adapter:
        model.enable_adapter_layers()
    else:
        model.disable_adapter_layers()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=0.1,
            do_sample=False,
        )
    return tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--start", type=int, default=1,
                        help="Resume from question N (1-indexed)")
    args = parser.parse_args()

    tokenizer, model = load_models()

    results = []
    base_hits_total = base_pts_total = 0
    tuned_hits_total = tuned_pts_total = 0

    SEP = "=" * 72
    DIV = "-" * 72

    questions_to_run = [q for q in QUESTIONS if q["id"] >= args.start]

    for q in questions_to_run:
        prompt = (
            f"VERBATIM PASSAGE:\nSource: {q['source']}\nRetrieved from: {q['retrieved_from']}\n\n"
            f"{q['passage']}\n\nQUESTION: {q['question']}"
        )

        print(SEP)
        print(f"TEST {q['id']}/30 — {q['source']}")
        print(SEP)
        print(f"PASSAGE:\n{q['passage']}\n")
        print(f"QUESTION:\n{q['question']}\n")

        # Base
        print(DIV)
        print("[ BASE QWEN 2.5-7B ]")
        print(DIV)
        base_ans = ask(prompt, tokenizer, model, use_adapter=False)
        print(base_ans)
        bh, bp = score_answer(base_ans, q["score_keywords"])
        print(f"\n  Score: {bh}/{bp}")
        base_hits_total += bh
        base_pts_total  += bp

        entry = {
            "id": q["id"],
            "source": q["source"],
            "correct_answer": q["correct_answer"],
            "base": {"answer": base_ans, "hits": bh, "points": bp},
        }

        if not args.base_only:
            print(DIV)
            print("[ TUNEDAI LABS — reasoning-tuned ]")
            print(DIV)
            tuned_ans = ask(prompt, tokenizer, model, use_adapter=True)
            print(tuned_ans)
            th, tp = score_answer(tuned_ans, q["score_keywords"])
            print(f"\n  Score: {th}/{tp}")
            tuned_hits_total += th
            tuned_pts_total  += tp
            entry["tuned"] = {"answer": tuned_ans, "hits": th, "points": tp}

        print(f"\n  CORRECT: {q['correct_answer']}\n")
        results.append(entry)

        # Save incrementally
        with open(OUTPUT_FILE, "w") as f:
            json.dump({
                "run_at": datetime.datetime.utcnow().isoformat() + "Z",
                "base_model": BASE_MODEL,
                "adapter": ADAPTER_REPO,
                "completed": len(results),
                "questions": results,
            }, f, indent=2)

    # Summary
    base_pct = round(100 * base_hits_total / base_pts_total, 1) if base_pts_total else 0
    print(SEP)
    print("FINAL — Raw Passage Reasoning Test (30 Questions)")
    print("Hume (1748) · Snow (1855) · Mill (1843) · Nightingale (1860)")
    print(SEP)
    print(f"Base Qwen 2.5-7B:           {base_hits_total}/{base_pts_total} = {base_pct}%")

    if not args.base_only and tuned_pts_total:
        tuned_pct = round(100 * tuned_hits_total / tuned_pts_total, 1)
        delta = round(tuned_pct - base_pct, 1)
        print(f"TunedAI Labs (tuned):        {tuned_hits_total}/{tuned_pts_total} = {tuned_pct}%")
        print(f"Delta:                       +{delta}%")

    print(SEP)
    print(f"Full results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
