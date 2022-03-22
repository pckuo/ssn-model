# ssn-model
Implementation of stabilized supralinear network (SSN) for sensory integration: normalization, contextual modulation, and dynamics

To support perception, neurons in sensory cortex integrate multiple inputs. Sensory integration is characterized by two neuronal response properties: (1) ‘‘normalization’’ — responses to multiple driving stimuli add sublinearly; and (2) surround suppression — modulatory contextual stimuli suppress responses to driving stimuli. Over the past decade, stabilized supralinear network (SSN) model — a rate-model neural network composed of excitatory and inhibitory neurons with power-law activation functions — was proposed as a unifying circuit for sensory integration. Their dynamics of multi-stability and oscillations may also underlie the circuit mechanism of working memory and rhythm generation.
In this project I implemented the SSN and examined its properties of normalization, surround suppression and dynamical structures. In the following sections, I will first introduce the formula and basic components of SSN, go through different model setups for normalization and surround suppression, and show examples of different dynamical structures of SSN.

Part I. 1-D ring model to demonstrate normalization and sublinear summation
1. To examine the normalization and sublinear summation property of SSN, a 1-D ring model (as in Ref. 2) was implemented to study the response of SSN to the superposition of
two drifting gratings of different orientations.
2. The model is a set of E/I pairs with varying preferred orientations. Preferred orientation is represented by the coordinate θ of an E/I pair on a ring. An oriented stimulus is presented the network to induce a Gaussian-shaped pattern of external input strengths peaked at the corresponding preferred orientation. For superposed gratings, the external inputs add linearly.
3. Nonlinear summation is demonstrated. For both excitatory and inhibitory firing rates, when stimulus is presented at 45°, the network response is the blue curve which peaks at 45°. Same result for stimulus at 135° is shown in orange curve. When stimuli are presented simultaneously at 45° and 135°, the network response is shown in green, which is smaller than the summation of the response of 45° and 135° individually.
4. The first part of the code (Part I: 1-D ring model for nonlinear summation) includes the model setup and simulations, which can be run to generate the above plots.

Part II. 1-D linear model to demonstrate surround suppression
1. To examine the property of surround suppression, or more general, context modulation, of SSN, a 1-D linear model (as in Ref. 2) was implemented to study the response of SSN to the interactions between stimuli in different visual positions, for example, in the classical receptive field and in the surround.
2. The mode is a 1D line of E/I pairs, with line position representing classical receptive field position in visual space. A drifting luminance grating induces a static external input, c*h(x), which has variable length (representing grating diameter) and peak height c.
3. When a narrow grating (length=1) was presented at x=0, the network response peaked at the corresponding CRF, hence the cells at x=0 has the highest response, for both E and I cells. When the length was increase to 5, as shown in orange, the E cell firing rate at x=0 drop significantly, much smaller than the surrounding. When the length is further increased to 7, as shown in green, the E cell firing is also lower than when length=1, and lower that the surrounding. What’s happening for the I cell firing rates is interesting as at length=5, the response is strong and reflects the shape of the input, which might underlie the strong decrease of the E cell firing rate. Whereas at length=7, the response of I cell at x=0 is also decreased, showing surround suppression.
4. The second part of the code (Part II: 1-D linear model for surround suppression) includes the model setup and simulations, which can be run to generate the above plots.

Part III. State space and stability analysis
1. To examine the underlying dynamical patterns of SSN, phase space analysis was performed using the simplest version of a single pair of E/I units to examine the steady states of the network, fixed points, and limit cycles.
2. Using the parameter regime demonstrated in Ref. 3, which gives the condition of the connection matrix W for the SSN to have different dynamic structures, two different dynamics were shown.
3. A steady state fixed point with nearby converging spiral dynamics. The firing rates oscillate and converge onto the steady state.
4. A limit cycle. Starting from the initial condition in the center, the trajectory travels outward, oscillates, and approaches the limit cycle.
5. The third part of the code (Part III: Phase space analysis of SSN) includes the implementation of the differential equations, which can be run to generate the above plots.

Reference:
1. Ahmadian Y, Rubin DB, Miller KD (2013) Analysis of the stabilized supralinear network. Neural Comput 25:1994–2037.
2. Rubin DB, Van Hooser SD, Miller KD (2015) The stabilized supralinear network: A unifying circuit motif underlying multi-input integration in sensory cortex. Neuron 85:402–417.
3. Kraynyukova N, Tchumatchenko T (2018). Stabilized supralinear network can give rise to bistable, oscillatory, and persistent activity. Proc Natl Acad Sci U S A. 2018 Mar 27;115(13):3464-3469.
