"""Titles, captions and 'how to read' notes for every standard figure.

Text may use inline HTML (<em>, <b>, <code>). Notation is defined where it is
first used; see also the glossary at the end of the report.
"""

EXPLAIN = {
    "binding_profiles": dict(
        section="input", title="Anchor binding profiles and loop lengths",
        caption="Binding profiles L and R along the region (left) and the lengths of the input loops (right).",
        explain="Chromatin loops are anchored by CTCF sites. <b>L</b> is the strength of sites that act as the "
                "<em>left</em> end of a loop, <b>R</b> the same for the <em>right</em> end, both scaled so their "
                "average is 1. Peaks mark likely loop anchors; the sampler places loop-extrusion factors (LEFs) "
                "where these profiles are high. The histogram shows how long the input loops are in kilobases "
                "(kb); the model later reproduces this length distribution."),
    "loop_matrices": dict(
        section="input", title="Loop maps J and J_loss",
        caption="Left: binary map J of loops and chain neighbours. Right: weighted loop map J_loss.",
        explain="Both are <em>bead &times; bead</em> matrices: a coloured pixel at (i, j) means a loop connects "
                "region positions i and j. <b>J</b> only records presence (1 or 0, the diagonal stripe is the chain "
                "itself). <b>J_loss</b> keeps the loop strengths and may be smoothed; it is used by the 2D data "
                "energy (<code>data_loss_mode = 1</code>) and as coupling for epigenetic states."),
    "tracks": dict(
        section="input", title="Input tracks",
        caption="Extra signal tracks resampled to simulation beads.",
        explain="Each panel is a genomic signal averaged per bead: BigWig tracks add binding energy, the LEF "
                "loading track tells where extruders prefer to start, and the compartment track (A positive, "
                "B negative) drives epigenetic states and the optional compartment force in 3D."),
    "energy_traces": dict(
        section="monte_carlo", title="Energy during the Monte Carlo run",
        caption="Total energy (left) and its binding, crossing and folding components (right) versus MC step.",
        explain="The sampler lowers a total energy E = binding + crossing + folding (arbitrary units, a.u.). "
                "<b>Binding</b> rewards loop ends on strong anchors; <b>crossing</b> penalises loops that "
                "cross or share anchors; <b>folding</b> rewards long loops. The grey zone is the <em>burn-in</em>, "
                "discarded before computing statistics. A healthy run falls quickly, then fluctuates around a "
                "stable level; a continuing downward slope means more steps are needed."),
    "convergence": dict(
        section="monte_carlo", title="Convergence diagnostics",
        caption="Autocorrelation of the folding energy (left) and unfolded fraction of the chain over time (right).",
        explain="Consecutive snapshots are correlated. The <em>autocorrelation</em> is 1 at lag 0 and decays "
                "as snapshots become independent; the dashed line marks the integrated autocorrelation time "
                "&tau; (MC steps between roughly independent samples). Increase <code>--sample-every</code> "
                "beyond &tau; to avoid redundant snapshots. The unfolded fraction is the share of the chain not "
                "covered by any loop: it should settle low once loops are established."),
    "lef_trajectories": dict(
        section="monte_carlo", title="Loop extrusion factor trajectories",
        caption="Every LEF as a coloured band between its two legs over the simulation.",
        explain="Horizontal axis is time (MC steps), vertical axis genomic position. A band's height is the "
                "size of that LEF's loop. Bands that grow and shift show extrusion and sliding; vertical breaks "
                "are unbinding and rebinding elsewhere (long jumps are masked). Stable bands at the same "
                "place indicate loops held by strong anchors."),
    "occupancy_lengths": dict(
        section="monte_carlo", title="LEF occupancy and loop lengths",
        caption="Average number of LEF legs per bead (left) and simulated loop-length distribution (right).",
        explain="Left: peaks show where LEFs accumulate; they should coincide with the anchors in the input "
                "profile. Right: loop lengths pooled over all LEFs and post-burn-in snapshots, in kb; compare "
                "with the input histogram to check the model produces realistic loop sizes."),
    "loop_joint": dict(
        section="monte_carlo", title="Joint distribution of loop ends",
        caption="Density of (left leg, right leg) positions of simulated loops.",
        explain="Each point is one loop (left end on the x axis, right end on the y axis, so everything lies "
                "above the diagonal). Dark spots are frequent loops between two specific anchors, which "
                "appear as dots or stripes similar to loop signals in a Hi-C map."),
    "mc_heatmap": dict(
        section="monte_carlo", title="Contact map from the loop ensemble",
        caption="Average contact frequency predicted from the loops alone (log colour scale).",
        explain="Every loop makes the beads inside it more likely to touch; averaging over snapshots gives a "
                "Hi-C-like map. Squares along the diagonal resemble TADs (topologically associating "
                "domains) and bright corner dots resemble loop signals."),
    "epi_trajectory": dict(
        section="monte_carlo", title="Epigenetic state trajectory",
        caption="Potts epigenetic state of each bead over the simulation.",
        explain="Beads carry discrete states (for example A-like, neutral, B-like). Horizontal stripes of one "
                "shade are stable domains; changes over time show domain formation and rearrangement."),
    "md_maps": dict(
        section="molecular_dynamics", title="Contact maps from 3D structures",
        caption="Mean contact probability over all MD frames (left) and its frame-to-frame standard deviation (right).",
        explain="Each frame converts bead distances d (nm) into a contact probability with a Gaussian kernel "
                "exp(&minus;d&sup2; / 2r&sup2;), where r is the contact radius. The mean map is the 3D analogue "
                "of a Hi-C map. The std map shows where contacts fluctuate most: bright pixels are flexible, "
                "dark ones are stable."),
    "md_series": dict(
        section="molecular_dynamics", title="Structure compaction and energy during MD",
        caption="Radius of gyration (left) and potential energy (right) versus MD time.",
        explain="The <em>radius of gyration</em> R<sub>g</sub> is the root-mean-square distance of beads from "
                "the centre of mass: smaller means a more compact fibre. Because loop constraints change from "
                "frame to frame the series fluctuates; a systematic drift at the start means the structure is "
                "still relaxing from its initial shape."),
    "mc_vs_md": dict(
        section="molecular_dynamics", title="Loop ensemble versus 3D structures",
        caption="Contact maps from MC loops (left) and MD (middle), and their decay with genomic separation (right).",
        explain="A consistency check between the two model stages. Maps are shown on a common 0-1 scale; r is "
                "the Pearson correlation of their off-diagonal values. P(s) is the mean contact at separation "
                "s beads (relative to s = 1) on log-log axes; similar slopes mean the 3D polymer reproduces "
                "the contact decay implied by the loops."),
    "validation_pearson": dict(
        section="validation", title="Validation: loop strength",
        caption="Experimental versus simulated loop strength.",
        explain="Per bead, the observed/expected (O/E) loop strength: loop weight divided by what is typical "
                "for that anchor distance, which removes the trivial fact that close anchors interact more. "
                "Both signals are log-transformed and z-scored. Top: experiment, bottom: simulation. Right: "
                "scatter on a 0-1 scale with the best-fit line; points on the dotted y = x line would agree "
                "perfectly. <b>Pearson r</b> measures linear and <b>Spearman &rho;</b> rank agreement."),
    "validation_pearson_direct": dict(
        section="validation", title="Validation: per-loop strength",
        caption="Experimental versus simulated strength of each individual loop (no bead aggregation).",
        explain="Same as the previous comparison but every loop is one point, which avoids averaging "
                "several loops that share an anchor bead. This is the most direct test of whether strong "
                "experimental loops are also strong in the model."),
    "validation_ps_decay": dict(
        section="validation", title="Validation: contact decay P(s)",
        caption="Experimental versus simulated contact strength as a function of bead separation.",
        explain="P(s) is the mean contact at separation s. Real chromatin loses contacts steadily with "
                "distance; agreement here shows the model produces realistic distance dependence."),
    "validation_insulation": dict(
        section="validation", title="Validation: boundary proxy",
        caption="Anchor density from the experiment versus the simulated insulation score.",
        explain="The insulation score is the mean contact across a bead (window of 5 beads each side): low "
                "values mark domain boundaries. Anchors often sit at boundaries, so a negative correlation is "
                "expected. This is only an approximate proxy, not a strict boundary validation."),
    "validation_summary": dict(
        section="validation", title="Summary of agreement",
        caption="Pearson r with 95% confidence intervals for every comparison.",
        explain="Each dot is a correlation; the bar is its 95% confidence interval (Fisher z-transform). "
                "Intervals that exclude 0 indicate a reliable association. Rough strength: |r| below 0.3 weak, "
                "0.3-0.5 moderate, 0.5-0.7 strong, above 0.7 very strong."),
}
