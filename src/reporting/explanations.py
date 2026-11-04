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
                "average is 1 (R is drawn mirrored below the axis so the two do not overlap). Peaks mark likely loop anchors; the sampler places loop-extrusion factors (LEFs) "
                "where these profiles are high. The histogram shows how long the input loops are in kilobases "
                "(kb); the model later reproduces this length distribution. <b>Used by the simulation</b> when <code>data_loss_mode = 0</code> (default): "
                "the data energy rewards LEF legs on high L (left leg) and high R (right leg); see the table <em>What the simulation uses as input</em>."),
    "anchor_map": dict(
        section="input", title="Input anchors: strength and CTCF orientation",
        caption="Loop arcs (top), anchor strengths with orientation (middle) and orientation confidence (bottom) along the region.",
        explain="<b>Top:</b> every input loop is an arc; height is its length, thickness its strength, colour its "
                "<em>configuration</em> from the CTCF motif directions of its two anchors. <b>Convergent</b> "
                "(&rarr; &larr;, the left anchor points right and the right anchor points left) is the classic "
                "loop-extrusion arrangement; tandem (&rarr; &rarr; or &larr; &larr;) and divergent (&larr; &rarr;) "
                "are less common; <b>uncertain</b> means an orientation is unknown or not trustworthy. "
                "<b>Middle:</b> one lollipop per anchor, height = strength, marker &gt; forward, &lt; reverse, "
                "&#9670; ambiguous, &#9679; unknown; larger markers are more confident calls. <b>Bottom:</b> "
                "orientation confidence = |2p &minus; 1|, where p is the probability that the motif is reverse "
                "(0 = coin flip, 1 = certain); calls below the dashed line are treated as uncertain."),
    "anchor_stats": dict(
        section="input", title="Anchor statistics",
        caption="Strength per orientation class (left), orientation confidence (middle) and loop configurations (right).",
        explain="Left: distribution of anchor strengths per orientation (box = middle 50 %, dots = individual anchors). "
                "Middle: histogram of orientation confidence; a peak near 1 means the orientation calls are "
                "reliable, a peak near 0 means they are close to guesses. Right: how many loops are convergent, "
                "tandem, divergent or uncertain, with their share. In real CTCF data convergent loops dominate; a "
                "different mix may mean noisy orientation calls or non-CTCF anchors."),
    "anchor_loops": dict(
        section="input", title="Loop strength, length and configuration",
        caption="Loop strength versus length (left) and the share of each configuration by loop-length bin (right).",
        explain="Left: each point is one loop (log length axis), coloured by configuration. Right: loops are "
                "split into four equal-sized length bins and the bar shows the share of each configuration in "
                "that bin; a falling convergent share for long loops would mean orientation is less informative "
                "at large distances."),
    "compartments": dict(
        section="input", title="Compartments (A/B)",
        caption="Signed compartment track with anchors (top), domain sizes (bottom left) and anchor strength per compartment (bottom right).",
        explain="<b>Top:</b> the compartment signal along the region, <b>A</b> (positive, red) is the active, "
                "open compartment and <b>B</b> (negative, blue-purple) the closed one; dotted lines mark where the sign "
                "changes (domain boundaries) and the dark strip at the bottom is the input anchor strength. "
                "<b>Bottom left:</b> how long A and B domains are in kb. <b>Bottom right:</b> input anchor strength "
                "on beads in A versus B (box = middle 50 %); the p-value is a Mann-Whitney test. In real data CTCF "
                "anchors often concentrate at compartment boundaries and in A; the compartment signal drives the "
                "epigenetic states and the optional compartment force in 3D."),
    "struct_time": dict(
        title="More structural metrics over time",
        caption="Shape asymmetry (left), scaling and fractal exponents (middle) and contacts per bead (right) along the simulation.",
        explain="Every metric is computed for each frame and plotted against MD time, so drifts (still relaxing) and fluctuations "
                "(a flexible structure) are visible. <b>Asphericity</b> and <b>acylindricity</b> are differences between the "
                "eigenvalues of the gyration tensor (the 3D spread of the beads along three principal axes): both 0 for a sphere. "
                "<b>&nu;</b> comes from R(s) ~ s<sup>&nu;</sup> (1/3 compact globule, 1/2 ideal chain); the "
                "<b>mass-radius fractal dimension</b> d<sub>f</sub> from N(r) ~ r<sup>d<sub>f</sub></sup> (beads within radius r; 3 fills space). "
                "<b>Contacts per bead</b> counts non-bonded neighbours within the contact radius."),
    "struct_bonds_angles": dict(
        title="Force check: bonds and angles",
        caption="Bond-length distribution (left), bend-angle distribution (middle) and bend angle over time (right), with thermal expectations.",
        explain="Adjacent beads are joined by a harmonic spring, so bond lengths should form a narrow bell around the rest length "
                "(dashed) with width &radic;(kT/k). The <b>angle force</b> penalises bending away from a straight chain (180&deg;). "
                "Its expectation (orange) is a Boltzmann curve including the geometric factor sin&theta;; a distribution that "
                "sits to the left of it means the force is too weak for the thermal noise, one far narrower means it is very stiff. "
                "A drifting angle over time means the structure has not equilibrated."),
    "struct_ev": dict(
        title="Force check: excluded volume",
        caption="Density of non-bonded pairs at short range (left), nearest-neighbour distances (middle), and overlapping pairs over time (right).",
        explain="Excluded volume stops beads passing through each other. Left: how many non-bonded pairs (at least 3 beads apart "
                "along the chain) are found at each distance, relative to large distances; a working repulsion makes it "
                "<em>drop towards zero</em> at small distances. Middle: nearest non-bonded neighbour of every bead. Right: share of "
                "non-bonded pairs closer than half a bond length; it should stay near zero. Larger values mean the repulsion is "
                "too soft or topoisomerase mode (excluded volume switched off on some beads) is active."),
    "struct_loops": dict(
        title="Loop distances in 3D",
        caption="Distance of loop anchors (left), their evolution (middle) and the 3D distance between neighbouring loops (right).",
        explain="Each loop is held by a spring between its two anchors; the distribution should sit near the rest length "
                "(dashed) if the loop force works, and the over-time curve should be flat after relaxation. Right: for two loops "
                "next to each other on the chain, the 3D distance between their facing anchors as a function of the genomic gap "
                "(line: median, band: middle half); a steep rise shows loops keep neighbours apart, a flat curve means loops cluster in space."),
    "struct_shape": dict(
        section="structure_3d", title="3D shape over time",
        caption="Size (left), shape (middle) and convex hull volume (right) of the structure along the trajectory.",
        explain="<b>R<sub>g</sub></b>, the radius of gyration, is the root-mean-square distance of beads from the "
                "centre of mass; <b>end-to-end</b> is the distance between the two chain ends (divided by 4 here "
                "to share the axis). <b>Shape anisotropy</b> &kappa;&sup2; comes from the eigenvalues of the "
                "gyration tensor: 0 is a perfect sphere, 1 a straight rod. <b>Hull sphericity</b> compares the "
                "convex hull (the smallest convex envelope of all beads) with a sphere of equal volume: 1 means "
                "spherical. <b>Hull volume</b> measures the space the chromatin occupies. Stable curves mean a "
                "stable overall fold; steady drifts mean the structures are still relaxing."),
    "optimizer": dict(
        section="optimizer", title="Optimizer convergence",
        caption="Loss of every short Monte Carlo evaluation (left) and how K, f, b and the temperature scale moved (right).",
        explain="The optimizer (Nelder-Mead) proposes the number of LEFs <b>K</b>, the folding coefficient <b>f</b> and the binding "
                "coefficient <b>b</b>, runs a short Monte Carlo for each, and scores the simulated loop map against the experimental "
                "loops (<b>loss</b> = 1 - correlation of the observed/expected maps + mismatch of the contact decay + 1 - correlation of the "
                "insulation scores; 0 is perfect). Every short run is a noisy estimate, so the best few candidates and the start are "
                "re-scored on longer runs with new seeds, and the winner is used for the production run. Dots: single evaluations; "
                "line: best so far; dashed and dotted lines: the long-run loss of the start and of the chosen parameters."),
    "optimizer_landscape": dict(
        section="optimizer", title="What the optimizer explored",
        caption="Loss against K (left), f (middle) and b (right) for every evaluation; later evaluations are darker, bigger dots mean a hotter temperature.",
        explain="Each dot is one short Monte Carlo evaluation. A good parameter shows a clear valley: low losses concentrated in a narrow range, "
                "with the dots getting darker (later) inside it. A flat cloud means the loss hardly depends on that parameter, or that the "
                "Monte Carlo noise is larger than the effect. The star marks the chosen parameters (its height is the long-run loss). "
                "Note that at a fixed temperature the energy depends on f/T and b/T only, so the temperature and the two coefficients partly "
                "trade off; several combinations can give the same loss.",
    ),
    "struct_snapshots": dict(
        section="structure_3d", title="Structure at the start, middle and end of the simulation",
        caption="Start, middle and end structures (pyvista), a filmstrip of 20 structures along the run, the Monte Carlo progress bar and the order parameter.",
        explain="Each render is an MD structure built from the loop configuration at that point of the Monte Carlo run "
                "(same camera, structures aligned, rainbow colour from the first bead to the last). <b>Filmstrip:</b> 20 small structures, each placed at the Monte Carlo step it was built from, so you can follow how the fold changes as the temperature changes; the boxed ones are the three representative structures A, B and C. <b>Monte Carlo bar:</b> burn-in (grey) is discarded "
                "from the statistics; the black ticks mark where the three structures come from. <b>Order parameter bar:</b> "
                "the Monte Carlo temperature T that controls how freely loops rearrange. It is constant for Metropolis and "
                "decays from T_init to T_final for Annealing, so the structure should become more ordered towards the end."),
    "struct_pairs": dict(
        section="structure_3d", title="Pair distances, packing and density",
        caption="Distribution of bead-bead distances (left), chain integrity and packing (middle), radial density (right).",
        explain="<b>Left:</b> histogram of the distances between every pair of non-neighbouring beads. "
                "<b>Middle:</b> the bond length should stay constant (chain integrity); the nearest non-bonded "
                "neighbour and the closest non-bonded pair show how tightly beads pack and whether any beads "
                "overlap (a closest pair far below the bond length would signal clashes). <b>Right:</b> bead "
                "density in spherical shells around the centre of mass; a plateau in the centre with a drop at "
                "the edge means a compact globule, a steady decline a loose coil."),
    "struct_scaling": dict(
        section="structure_3d", title="Scaling, fractal dimension and structural change",
        caption="Polymer scaling R(s) (left), fractal dimension per frame (middle) and frame-to-frame RMSD (right).",
        explain="<b>R(s)</b> is the mean 3D distance between beads s apart; its log-log slope is the exponent "
                "&nu;. <b>Fractal dimension</b> d<sub>f</sub> says how the structure fills space: from scaling "
                "d<sub>f</sub> = 1/&nu; (3 for a compact globule, 2 for an ideal chain, about 1.7 for a swollen "
                "chain), and independently from the <em>mass-radius</em> relation N(r) &prop; r<sup>d<sub>f</sub></sup>, "
                "the number of beads within distance r of the centre. Agreement of the two supports the estimate. "
                "<b>RMSD</b> (root-mean-square deviation) after optimal rotation measures how much the structure "
                "changes between consecutive frames."),
    "struct_compartments": dict(
        section="structure_3d", title="Compartments in 3D",
        caption="Radial position (left) and compactness (right) of A and B beads along the trajectory.",
        explain="Beads are split by the sign of the compartment signal (A &gt; 0 red, B &lt; 0 blue-purple). Left: mean "
                "distance of each group from the centre of mass; larger means closer to the surface. Right: the "
                "radius of gyration of each group alone. In real nuclei B (closed) chromatin tends to be more "
                "compact and sit at the periphery, A more central."),
    "viewer_3d": dict(section="structure_3d", title="Interactive 3D viewer", caption="", explain=""),
    "loop_matrices": dict(
        section="input", title="Loop maps J and J_loss",
        caption="Left: binary map J of loops and chain neighbours. Right: weighted loop map J_loss.",
        explain="Both are <em>bead &times; bead</em> matrices: a coloured pixel at (i, j) means a loop connects "
                "region positions i and j. <b>J</b> only records presence (1 or 0, the diagonal stripe is the chain "
                "itself). <b>J_loss</b> keeps the loop strengths and may be smoothed; it is used by the 2D data "
                "energy (<code>data_loss_mode = 1</code>; with the default <code>0</code> the simulation uses L and R instead) and J is the coupling for "
                "epigenetic states (only when <code>epi_coeff &gt; 0</code>). J_loss is smoothed only if input smoothing is on (Gaussian, "
                "sigma = 1% of the beads), so by default it is the raw loop strength; see the table <em>What the simulation uses as input</em>."),
    "tracks": dict(
        section="input", title="Input tracks",
        caption="Extra signal tracks resampled to simulation beads.",
        explain="Each panel is a genomic signal averaged per bead (line = signal, shaded = area under it). BigWig "
                "tracks add binding energy; the LEF loading track tells where extruders prefer to start. The "
                "title of each panel gives its Spearman correlation &rho; with the input anchor strength (smoothed): "
                "positive means the track is high where anchors are strong."),
    "energy_traces": dict(
        section="monte_carlo", title="Energy during the Monte Carlo run",
        caption="Total energy with its plateau (left) and component fluctuations around their plateaus (right) versus MC step.",
        explain="Left: total energy with its post-burn-in plateau (dashed line, shaded band = &plusmn;1 standard deviation). "
                "Right: each component minus its own plateau, so fluctuations are comparable even though the "
                "components differ by orders of magnitude. The sampler lowers a total energy E = binding + crossing + folding (arbitrary units, a.u.). "
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
        section="monte_carlo", title="LEF trajectories",
        caption="Every loop extrusion factor as a coloured band between its two legs over the simulation.",
        explain="Horizontal axis is time (MC steps), vertical axis genomic position (Mb). Each coloured band is "
                "one LEF: its lower edge is the left leg, its upper edge the right leg, so the band height is the "
                "loop size and the shaded area is the chromatin the LEF holds in a loop. Colour encodes the LEF index "
                "(0, 1, 2, ...), so you can follow one LEF over time. Bands that grow and shift show extrusion and "
                "sliding; gaps are unbinding and rebinding elsewhere (the jump itself is not drawn); flat bands are "
                "loops held by strong anchors. Each stretch between rebindings is lightly smoothed in time; a random "
                "subset is drawn when there are many LEFs."),
    "replicas": dict(
        title="Replica agreement",
        caption="Energy of each independent chain (left), loop occupancy along the region (middle) and their correlation (right).",
        explain="Replicas are independent Monte Carlo chains started from different random states. If the sampler has converged they "
                "should tell the same story. Left: smoothed energy after burn-in; overlapping curves are good. "
                "<b>R-hat</b> (Gelman-Rubin) compares the variation between chains with the variation inside chains: values close to 1 "
                "(below about 1.1) mean the chains agree. Middle: one row per replica, bright where many loops cover a bead (each row scaled to its own maximum); matching stripes mean the same loop pattern was found. Right: Pearson correlation "
                "of these profiles for every pair of replicas, 1 meaning identical. Low values warn that the result depends on the "
                "random start, so run longer or check the energy parameters. Downstream maps and 3D structures use all replicas together."),
    "kymograph_zoom": dict(
        section="monte_carlo", title="Kymograph zoom near equilibrium",
        caption="A short window of the production phase, shown without time averaging.",
        explain="With many MC steps the full kymograph averages neighbouring snapshots and single events blur together. Left: the whole "
                "run with the chosen window shaded (by default the middle of the production phase, where the system is equilibrated). "
                "Middle: loop coverage in that window. Right: positions of the LEF legs. Horizontal streaks are loops that stay put; "
                "diagonal or short vertical steps are legs sliding or loops jumping to a new anchor. Choose another window with "
                "<code>zoom-start</code> and <code>zoom-snapshots</code>."),
    "kymograph": dict(
        section="monte_carlo", title="Loop dynamics kymograph",
        caption="Loop coverage (left, violet) and LEF-leg density (middle, teal) over time, with the time-averaged coverage (right).",
        explain="A <em>kymograph</em> shows position (vertical, Mb) against time (horizontal, MC steps), smoothed for "
                "readability. <b>Left:</b> how many loops cover each position; bright bands are regions held inside "
                "loops. <b>Middle:</b> where the LEF legs are; sharp horizontal lines are legs parked on anchors, "
                "diagonal streaks are legs sliding. <b>Right:</b> coverage averaged over the post-burn-in time "
                "(violet) next to the input anchor strength (orange). The dashed line marks the end of burn-in."),
    "occupancy_lengths": dict(
        section="monte_carlo", title="LEF occupancy and loop lengths",
        caption="Occupancy against anchor strength (left), occupancy around strong anchors (middle), loop lengths (right).",
        explain="Left: every dot is one bead; x is the input anchor strength (L+R), y how often a LEF leg sits on that bead; "
                "colour is the position along the region. If the model follows the data, dots rise to the right; the black line "
                "is the median in anchor-strength bins and <b>rho</b> the Spearman rank correlation of the smoothed profiles. "
                "Middle: average LEF occupancy in a window around strong anchors (local maxima of the input signal above its "
                "75th percentile, violet) compared with the same number of random positions (grey band, 5-95%). A violet peak at 0 "
                "standing above the band shows LEFs really stop at the anchors. Right: loop lengths (kb) pooled over LEFs and "
                "post-burn-in snapshots (filled) against the input loops (black outline); <b>KS</b> is the Kolmogorov-Smirnov "
                "distance (0 = identical, 1 = disjoint)."),
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
    "epigenetic_maps": dict(
        section="monte_carlo", title="Epigenetic model: loop map, checkerboard and combined map",
        caption="Loop contact map, epigenetic checkerboard and their combination.",
        explain="Left: the contact map from loops alone. Middle: the <b>epigenetic checkerboard</b>, the outer product "
                "&lang;s<sub>i</sub>&rang;&lang;s<sub>j</sub>&rang; of the average epigenetic state of every bead (scaled to &minus;1 B-like ... +1 A-like): "
                "red where both beads are A-like or both B-like (same type), blue where one is A-like and the other B-like. "
                "Right: the <b>combined map</b> = loop map &times; exp(0.5 &middot; checkerboard). The exponential keeps contacts positive; "
                "same-type pairs are enriched up to about 1.6&times; and mixed pairs depleted to about 0.6&times;, so the plaid pattern is a weaker "
                "modulation on top of the loops. This combined map is the one used for validation against data."),
    "epigenetic_features": dict(
        section="monte_carlo", title="Epigenetic model: effect on compartments, insulation and decay",
        caption="Compartment signal, insulation score and P(s) of the loop-only and combined maps.",
        explain="Left: PC1 of the combined map (red = A-like, blue-purple = B-like) with the average epigenetic state (black); they should agree "
                "when the checkerboard drives the compartments. Middle: insulation score, loops only (grey) versus combined (violet); the two should keep "
                "boundaries at the same places because the checkerboard is a weaker modulation. Right: diagonal decay P(s) relative to s = 1."),
    "loop_nesting": dict(
        section="monte_carlo", title="Loop nesting over time",
        caption="Left: loops by nesting order over the run. Middle: order distribution early, middle and late. Right: mean and deepest order.",
        explain="Loops can sit inside each other. A <b>parent</b> is a loop that holds other loops but is inside none, while an <b>orphan</b> is isolated: inside no loop and holding none. If A contains B, then B is a "
                "<b>child of order 1</b>; if A contains B and B contains C, then C has <b>order 2</b>, and so on. The order of a loop is simply how many "
                "other loops enclose it. <b>Left:</b> stacked loop counts per order for the first chain, burn-in included (dashed line = end of burn-in). "
                "<b>Middle:</b> fraction of loops of each class (orphan, parent, order 1, 2, 3+) in the first, second and last third of the production run; a shift to the right means "
                "loops nest deeper. <b>Right:</b> mean order of all loops and the deepest order in each snapshot. Identical loops count once."),
    "loop_parents_children": dict(
        section="monte_carlo", title="Parents and children",
        caption="Left: parent length vs loops inside it. Middle: length per order. Right: child length relative to its parent, over time.",
        explain="<b>Left:</b> each dot is a parent; the y-axis counts all loops inside it (any order). Longer parents usually hold more loops. "
                "<b>Middle:</b> distribution of loop length for orphans, parents and children of order 1, 2, 3+ (the legend gives the mean). Children are "
                "expected to be shorter, since they fit inside. <b>Right:</b> ratio of a child's length to that of its smallest enclosing loop, "
                "in the early, middle and late third of production; values near 1 mean a child nearly fills its parent."),
    "motif_search": dict(
        section="input", title="CTCF motif search",
        caption="Left: loops after each filtering step. Middle: best motif score per anchor against the cutoff. Right: orientation of the left vs right anchor.",
        explain="LoopSage can read the orientation of CTCF at every loop anchor straight from the reference genome (<code>--motif-genome</code>). "
                "<b>How it works:</b> the anchor sequence is scanned on both strands with the CTCF motif (a table of how likely each base is at each of "
                "the 19 motif positions). A window gets a <b>log-odds score</b>: higher means closer to the ideal CTCF site. It counts as a hit if its "
                "score beats the <b>cutoff</b>, set so that random DNA reaches it with probability <code>--motif-pvalue</code>. <b>Orientation:</b> "
                "P(reverse) is the share of hit strength on the reverse strand (each hit weighted by 2<sup>score</sup>, so the strongest site dominates); "
                "0 = forward, 1 = reverse, no motif = unknown. <b>Left:</b> loops surviving each step (region, minimum PET count <code>--min-pet</code>, "
                "oriented anchors). <b>Middle:</b> best score of each anchor; hits sit to the right of the dashed cutoff. <b>Right:</b> for loops with both "
                "anchors oriented, P(reverse) of the left anchor against the right one. <i>Convergent</i> loops (forward on the left, reverse on the right, "
                "top-left corner) are the ones loop extrusion produces, so a strong top-left corner is the expected signature."),
    "lef_speed_init": dict(
        section="monte_carlo", title="Initial LEF speeds",
        caption="Speeds given to the LEFs at the start, and how they share the sliding moves.",
        explain="Every LEF gets a <b>speed</b> v at the start (relative to the default sliding rate, so the mean is always 1): its right leg moves at "
                "+v and its left leg at &minus;v, so the loop opens at 2v. In the default model all LEFs are equal (a single bar at 1). "
                "Choose a distribution with <code>--lef-speed</code> (or <code>lef_speed_dist</code>): <i>maxwell</i> is the Maxwell-Boltzmann speed "
                "distribution, what a gas of particles at thermal equilibrium has; <i>lognormal</i> (width <code>--lef-speed-spread</code>) is a broad "
                "distribution with a few very fast LEFs; <i>two_state</i> mixes slow and fast LEFs (<code>--lef-fast-fraction</code>, "
                "<code>--lef-fast-ratio</code>). Left: the speeds in this run (bars) and the theoretical curve. Right: a Lorenz curve, the share of all "
                "sliding moves done by the slowest LEFs; the dashed diagonal is the case where all LEFs are equal, the more it bends the more unequal "
                "the LEFs. A speed changes how <i>fast</i> a LEF explores, not where it ends up: the equilibrium loops stay the same."),
    "lef_velocity": dict(
        section="monte_carlo", title="LEF speed statistics",
        caption="Speed distribution, mean squared change of loop length, and velocity autocorrelation.",
        explain="Between two snapshots a LEF changes its loop length by &Delta;&ell;. Its <b>speed</b> is v = |&Delta;&ell;| / (2 &Delta;t) "
                "(each leg moves at &plusmn;v; always an absolute value; rebinding jumps are excluded). Left: the distribution <b>P(v)</b> on a log "
                "scale against a half-Gaussian with the same mean square speed, the one-dimensional Maxwell-Boltzmann shape of a thermal gas. "
                "A tall spike at zero means many LEFs are parked; heavy tails mean a mixture of fast and slow LEFs. Middle: the <b>mean squared "
                "displacement</b> of the loop length, MSD(t) ~ t<sup>a</sup>: a = 1 is ordinary diffusion (random-walk sliding), a = 2 is ballistic motion "
                "(steady extrusion) and a &lt; 1 means the loop is trapped. With set speeds, each speed level is a separate curve; faster LEFs "
                "should explore faster at short times. Right: the <b>velocity autocorrelation</b> C(t) = &lang;v(0) v(t)&rang; / &lang;v&sup2;&rang; of "
                "the signed length change; positive means motion persists (extrusion), negative means the loop is pulled back (confinement)."),
    "lef_phase_space": dict(
        section="monte_carlo", title="LEF phase space and free energy of the loop length",
        caption="Free energy of (length, speed), free energy of the length with the mean speed, and kinetic temperature near anchors.",
        explain="A <b>phase space</b> has a coordinate on one axis and its speed on the other. Left: the <b>free energy</b> F = &minus;ln P in units "
                "of the effective <i>kT</i>, where P is how often a LEF is found at that loop length and speed (dark = most probable, F = 0; yellow = rare, blank = never "
                "seen); the teal line is the mean speed at each length. A dark band near v = 0 is the parked population. Middle: the free energy of the "
                "loop length alone, F(&ell;), as a line, with the mean speed (black): valleys are preferred loop lengths, and speed usually drops inside them. "
                "Right: the <b>kinetic temperature</b> &lang;v&sup2;&rang; (relative to its mean) against the distance of the nearest loop leg to a strong "
                "anchor; below 1 near zero distance means anchors slow LEFs down (blocking), a flat line means anchors do not affect speed."),
    "lef_free_energy": dict(
        section="monte_carlo", title="Free energy of loop position and of the two legs",
        caption="F(midpoint, length) with the flow, and F(left leg, right leg).",
        explain="The same idea, F = &minus;ln P (dark = most probable), in other variables. Left: F over loop midpoint and length, with teal arrows for the "
                "mean motion of LEFs in each cell (arrow length ~ speed; these arrows keep their sign, they show where loops go). Arrows converging on dark "
                "cells show loops funnelled into stable states, upward arrows mean loops are still growing. Right: F over the positions of the two legs, "
                "the natural landscape of a LEF; dark spots are leg pairs that are held together for a long time (anchor pairs), pink ticks mark the "
                "strong anchors."),
    "lef_free_energy_3d": dict(
        section="monte_carlo", title="Free-energy surface of loop length and speed (interactive)",
        caption="F(length, speed) as a 3D surface; drag to rotate, scroll to zoom.",
        explain="The free energy F = &minus;ln P of the phase plane (loop length, speed) drawn as a <b>surface</b>: the height is F in units of the effective "
                "<i>kT</i>. Wells are the probable states (F = 0 is the most probable), hills are rare ones (capped at 7 <i>kT</i> where a state is practically "
                "never seen). The contour map on the floor shows the same surface from above and the teal line there is the mean speed at each length. "
                "A well elongated along the speed axis means loops of that length move at many speeds; a well pinned at low speed means they are parked. "
                "In the report you can drag to rotate the surface, scroll to zoom and hover to read the values."),
    "epi_trajectory": dict(
        section="monte_carlo", title="Epigenetic states over time",
        caption="Epigenetic state of every bead during the simulation (epigenetic model only).",
        explain="Each bead carries one of a few epigenetic states, for example active (A-like), neutral and inactive "
                "(B-like) chromatin. Colours run from blue-purple (B-like, closed) to red (A-like, active; with a compartment track the high states are pulled towards A regions); bands of one colour are stable epigenetic domains; a band that changes "
                "colour is a domain that formed, merged or was lost. Beads joined by the backbone or by a loop prefer the "
                "same state, so domains tend to follow the loops."),
    "epigenetic_model": dict(
        section="monte_carlo", title="Epigenetic model: domains and energy",
        caption="Average epigenetic state, model energy over time and the chance that two beads share a state.",
        explain="Left: the <b>average epigenetic state</b> of each bead over the production snapshots (violet line) with its "
                "spread over time (shaded). A flat shaded band means the bead keeps one state; a wide band means it keeps "
                "switching. The shading marks the B-like (blue-purple) and A-like (red) halves of the state range, and the black line is the input compartment signal rescaled to the same range, so you can see whether "
                "the simulated domains follow it. Middle: the <b>epigenetic energy</b> over the run; it should fall and then "
                "fluctuate around a plateau, the dashed line marks the end of burn-in. Right: the probability that two beads "
                "are in the <b>same epigenetic state</b>; bright squares on the diagonal are epigenetic domains, bright "
                "off-diagonal patches are distant regions of the same type (like A/B compartments)."),
    "epigenetic_loops": dict(
        section="monte_carlo", title="Epigenetic model: loops and epigenetic state",
        caption="Do loop ends share an epigenetic state, and how large is each state's share over time?",
        explain="Left: the chance that the two ends of a loop are in the same epigenetic state (violet) compared with any two "
                "beads at the same genomic distance (grey). Violet above grey means loops bring together chromatin of the "
                "same type, the signature of loops organising epigenetic domains; equal curves mean no link. Right: the "
                "fraction of beads in each state during the run (blue-purple = B-like, red = A-like). Curves that flatten show the state populations have "
                "equilibrated; a state that disappears means one epigenetic type dominates the region."),
    "mc_features": dict(
        title="Loop ensemble: map, compartments and insulation",
        explain="Left: the contact map implied by the simulated loops; triangles on the diagonal mark TAD-like boundaries. "
                "Top right: the <b>insulation score</b>, the mean contact across each "
                "bead inside a diamond; valleys mean few contacts cross that position, i.e. a domain boundary (it shares the genomic axis of the map). "
                "Bottom right: the <b>diagonal decay P(s)</b>, the mean contact at bead separation s (log-log); its slope shows how fast contacts fade "
                "with distance, and bumps mark typical loop lengths. Each loop gives a smooth dot, a weaker filled triangle below it, and, through "
                "chains of loops (i-j and j-k also enrich i-k), progressively weaker dots; stripes arise from moving LEF legs, they are not drawn in."),
    "md_features": dict(
        title="3D structures: map, compartments and insulation",
        explain="Same analysis as for the loop ensemble, now on the contact map averaged over the MD structures: heat map "
                "(left), insulation score with boundaries (top right) and diagonal decay P(s) (bottom right). "
                "Compare it with the MC version to see which features survive when loops are turned into a 3D polymer."),
    "features_compare": dict(
        explain="MC (violet), MD (green) and, when available, the experiment (black dashed) overlaid. Left: compartment signals against the input track: for MC the average epigenetic state of every bead (only when the epigenetic model is on), "
                "for MD the PC1 of the contact map (sign-aligned, since an eigenvector's sign is arbitrary); the title gives the MC-vs-MD |r| and the legend the r with the data. Middle: insulation score as z-scores (the maps differ in contrast), valleys at the same positions mean the same domain boundaries. "
                "Right: <b>P(s)</b>, the mean contact at genomic separation s (a diagonal of the map) on log-log axes; "
                "the slope is the decay exponent, the closer the lines the better the 3D polymer reproduces the loop ensemble; the experiment is computed "
                "from the input loops (no inference) and r in the legend is against it."),
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
    "md_geometry": dict(
        section="molecular_dynamics", title="3D geometry: scaling and loop proximity",
        caption="Mean 3D distance versus genomic separation (left) and spatial distance of loop anchors versus random pairs (right).",
        explain="<b>R(s)</b> is the average 3D distance (nm) between two beads s beads apart. On log-log axes it follows "
                "R ~ s<sup>&nu;</sup>: &nu; = 1/3 is a compact globule, 1/2 an ideal chain, about 0.59 a swollen chain. "
                "Right: anchors of simulated loops are compared with random bead pairs at the <em>same</em> genomic "
                "separation. If loops work as intended, the anchor distances sit clearly to the left (closer) of the "
                "random ones; the gap between the medians measures how much loop extrusion compacts the chromatin."),
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
        section="validation", title="Validation: insulation score",
        caption="Insulation score of the experimental loop map versus the simulated contact map, with domain boundaries.",
        explain="The <b>insulation score</b> of a bead is the mean contact between the 5 beads on its left and the 5 on its right: low values mean "
                "little contact crosses that bead, i.e. a domain (TAD-like) <b>boundary</b>. It is computed in the same way on the experimental loop "
                "map (input loops, exactly as given, no inference) and on the simulated map. <b>Top:</b> both scores as z-scores; triangles mark "
                "boundaries (local minima). <b>Bottom:</b> where the simulation is above or below the experiment. <b>Right:</b> correlation, and "
                "the share of experimental boundaries that also appear in the simulation within 2 beads. A high correlation means the model puts "
                "domain boundaries where the data do."),
    "validation_insulation_anchor": dict(
        section="validation", title="Validation: insulation vs anchor density",
        caption="Experimental anchor density versus the simulated insulation score.",
        explain="Anchors often sit at domain boundaries, so the simulated insulation score is compared with the density of input anchors per bead. "
                "A negative correlation is expected (more anchors, lower insulation). It is a looser check than the insulation score of the experimental loop map."),
    "validation_pc1": dict(
        section="validation", title="Validation: compartments",
        caption="Input compartment track versus the simulated compartment signal: average epigenetic state (MC) or PC1 of the contact map (MD).",
        explain="The simulated compartment signal is compared, position by position, with the compartment track you supplied. <b>MC:</b> the signal is the "
                "<b>average epigenetic state</b> of each bead over the production run (A-like positive, B-like negative); it exists only when the epigenetic "
                "model is on, otherwise this check is skipped. <b>MD:</b> the signal is <b>PC1</b>, the first principal component of the observed/expected "
                "correlation map of the 3D contacts (its sign is arbitrary, so it is aligned to the track). A high correlation means the simulation reproduces the compartment pattern of the data."),
    "validation_summary": dict(
        section="validation", title="Summary of agreement",
        caption="Pearson r with 95% confidence intervals for every comparison.",
        explain="Each dot is a correlation (violet = loop ensemble MC, green = 3D structures MD, smoothed where available; number = r); the bar is its 95% confidence interval (Fisher z-transform); shaded bands mark weak (0-0.3) and strong (0.5-0.7) agreement. "
                "Intervals that exclude 0 indicate a reliable association. Rough strength: |r| below 0.3 weak, "
                "0.3-0.5 moderate, 0.5-0.7 strong, above 0.7 very strong."),
}
