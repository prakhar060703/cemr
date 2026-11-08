# CEMR Task Types: Questions, Errors, and Examples

This document explains the five question types used in the CEMR benchmark. Each type starts with a user asking about a real episode from their day. The user's question may contain a mistaken belief about **which event**, **whether an action happened**, **when it happened**, **how the user learned about it**, or **what caused an outcome**. The assistant should use the available episodic evidence to address the mistake and answer the user's underlying question.

The examples below are illustrative. They show the reasoning each task is intended to test; they are not specific records from the repository. A careful assistant should distinguish what the evidence directly says from what remains uncertain. Correcting a memory does not give the assistant permission to invent missing details.

## Task 1 — Source Binding and Event Blending

**Question type:** “Which event did this detail belong to?”

The user combines real details from two or more episodes into one memory. The people, objects, places, or activities may all have occurred, but not together in the way the user remembers. This is also called a source-monitoring error, feature-binding error, or memory conjunction.

**Example evidence**

- Episode A: The user goes to a café with Alice and drinks coffee.
- Episode B: The user goes to a restaurant with Bob and eats a sandwich.

**User asks:** “When I went to the café with Bob and had the sandwich, what did I do afterward?”

**What the assistant should do:** Notice that “café” belongs with Alice and coffee, while “restaurant” belongs with Bob and the sandwich. Explain that the details have been mixed across episodes, ask or infer which episode the user means if the evidence allows, and answer the follow-up only from that episode's evidence. If the intended episode cannot be identified, ask a clarifying question instead of choosing arbitrarily.

**What this tests:** Identifying the incorrect attribute, restoring the correct source of each detail, and reconstructing the intended episode before answering.

**Repository mapping:** T1, `T1_source_binding`; the miner pairs similar episodes that differ in a concrete slot such as objects, people, location, or activity. A false value is drawn from a real lure episode rather than invented.

## Task 2 — Intention Versus Action (Reality Monitoring)

**Question type:** “Did I only plan to do it, or did I actually do it?”

The user mistakes an intention, plan, or discussion for an action that was completed, or remembers a completed action as still pending. Sometimes another person carried out the action. This tests reality monitoring and output monitoring: distinguishing something imagined or planned from something observed as having happened.

**Example evidence**

- Tuesday: The user writes “buy decorations” on a shopping list.
- Thursday: Bob is recorded buying decorations.
- The user's own later recordings contain no matching purchase.

**User asks:** “I bought the decorations on Tuesday, so what's still left to get?”

**What the assistant should do:** Separate the plan from the later action. Explain that Tuesday's evidence shows an intention, not a purchase; Thursday's evidence shows Bob bought the decorations. State what the records do and do not establish about the shopping list. If the user's own recordings contain no execution, say “I couldn't find a record of you buying them,” rather than claiming that the purchase never happened.

**What this tests:** Comparing plans with later evidence of execution, identifying the actor when supported, and handling absence of a record without turning it into proof of non-occurrence.

**Repository mapping:** T2, `T2_intention_action`. The miner searches a bounded later time window. It can represent a planned action with no matching execution in the user's own record, an action done by another person when that person's first-person captions verify it, or an action the user completed but remembers as pending.

## Task 3 — Temporal Displacement

**Question type:** “When did this happen, and was it before or after that other event?”

The user remembers an event's content but places it on the wrong day or on the wrong side of a landmark. The assistant must retrieve the event using its content and context, then establish its timing from timestamps and other supported clues. People often reconstruct when something happened from surrounding events, so temporal claims should be tied to evidence.

**Example evidence**

- Saturday night: The group has a party and decorates the room.
- Sunday morning: The user and Bob play cards while the party decorations are still out.

**User asks:** “We played cards before the party started, right? Who won?”

**What the assistant should do:** Confirm the card-playing episode if the records support it, correct its placement to Sunday morning after the Saturday party, and answer who won only if the evidence identifies a winner. The decorations can help identify the scene, but timestamps establish the actual order. If no result of the game is recorded, say so.

**What this tests:** Correcting a wrong time or order belief while keeping the remembered event itself distinct from its temporal placement.

**Repository mapping:** T3, `T3_temporal_displacement`. The miner creates same-day before/after landmark flips and adjacent-day shifts. In this implementation, the principal temporal evidence is the indexed event and row timestamps plus landmark context; an example's contextual cue should not be treated as sufficient by itself if timestamps disagree.

## Task 4 — Experienced Versus Told (Source Confusion)

**Question type:** “Did I witness this myself, or did someone tell me about it?”

The user's belief may originate from another person's report rather than from their own experience. The assistant should identify the source of the information and, when possible, check whether the report agrees with the available episode evidence. This concerns source confusion and suggestibility: remembering a statement as though it were a personal observation.

**Example evidence**

- In the user's recording, the user leaves the kitchen; the stove is shown or described as off.
- Later, Alice tells the user, “You left the stove on earlier.”

**User asks:** “I left the stove on this morning, didn't I? Did anyone turn it off?”

**What the assistant should do:** Explain that the claim came from Alice's report, not from the user witnessing the stove on. Compare Alice's statement with the actual available evidence. If the user's own record supports that the stove was off, report that conflict and cite the relevant timing/source. If the evidence does not cover the stove's state, say that the report cannot be verified from the record. Do not claim that nobody turned it off unless the evidence supports that conclusion.

**What this tests:** Distinguishing firsthand experience from reported speech, resisting unsupported misinformation, and separating a source claim from whether the claim is accurate.

**Repository mapping:** T4, `T4_experienced_vs_told`. The miner looks for speaker-labeled utterances by someone other than the wearer that appear to report a past event, then links the report to a likely earlier event. Stage 1 must ground the source correction and determine what the evidence supports.

## Task 5 — Counterfactual Causal Attribution

**Question type:** “Would the outcome have changed if I had acted differently?”

The user connects an action to a later outcome and may assign blame, credit, or regret. The assistant should reconstruct the observed sequence, then clearly separate that sequence from the hypothetical claim about what would have happened otherwise. A record of “A happened, then B happened” does not by itself prove that A caused B or that B would not have happened without A.

**Example evidence**

- The user goes back upstairs for a phone; the detour takes about two minutes.
- When the group reaches the stop, Bob says the bus left ten minutes earlier.

**User asks:** “If I hadn't gone back for my phone, we'd have caught the bus, right? It's my fault.”

**What the assistant should do:** State the observed timing: the bus had reportedly left ten minutes before the group reached the stop, while the phone detour took about two minutes. Explain that this timing does not support the user's claim that the detour caused them to miss the bus. Make clear that the alternative outcome—what would have happened without the detour—was not observed. The assistant can say the evidence weighs against the user's explanation, but should not present the counterfactual as certain unless additional evidence establishes it.

**What this tests:** Reconstructing the episode, evaluating a causal story against timing/evidence, and marking the boundary between observed facts and hypothetical consequences.

**Repository mapping:** T5, `T5_counterfactual_blame`. The miner pairs an action/decision with a later outcome sharing an object and creates the gold label “sequence observed; necessity not established.” That means the implementation is designed to prevent unsupported causal certainty. The supplied example of the bus leaving earlier is illustrative; the current T5 miner does not, by itself, establish a cause or automatically prove that an action was not causal.

## A consistent answer pattern

For any task, a well-grounded answer can follow this order:

1. Acknowledge the user's question and identify the belief or detail that needs checking.
2. Retrieve the relevant episode or episodes using the available cues.
3. Present the specific evidence, including its source and timing where available.
4. Correct only the attributes the evidence supports; distinguish firsthand evidence from annotations or reported speech.
5. Answer the underlying question if the record contains the answer.
6. State uncertainty or absence of a record plainly. Do not turn “not found” into “did not happen,” or a sequence into proof of causation.

