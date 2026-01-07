"""Physics topic sampling for question generation."""

import random
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..models.schemas import PhysicsTopic


@dataclass
class TopicContext:
    """Context for generating a physics question."""

    topic: PhysicsTopic
    subtopic: str
    difficulty_modifier: str
    additional_context: Optional[str] = None

    def to_prompt_string(self) -> str:
        """Convert to a string for use in prompts."""
        base = f"{self.subtopic} in {self.topic.value}"
        if self.difficulty_modifier:
            base += f", {self.difficulty_modifier}"
        if self.additional_context:
            base += f". {self.additional_context}"
        return base


class TopicSampler:
    """Samples physics topics and subtopics for question generation."""

    # Comprehensive mapping of topics to graduate-level subtopics
    SUBTOPICS: Dict[PhysicsTopic, List[str]] = {
        PhysicsTopic.CLASSICAL_MECHANICS: [
            "Lagrangian mechanics and variational principles",
            "Hamiltonian mechanics and phase space",
            "Central force problems and Kepler orbits",
            "Rigid body dynamics and Euler's equations",
            "Coupled oscillators and normal modes",
            "Canonical transformations and generating functions",
            "Hamilton-Jacobi theory",
            "Action-angle variables",
            "Small oscillations and stability analysis",
            "Nonlinear dynamics and chaos",
            "Noether's theorem and conservation laws",
            "Poisson brackets and symplectic structure",
        ],
        PhysicsTopic.ELECTROMAGNETISM: [
            "Maxwell's equations in matter",
            "Electromagnetic waves in dispersive media",
            "Waveguides and cavity resonators",
            "Radiation from accelerated charges",
            "Multipole expansion of electromagnetic fields",
            "Relativistic electrodynamics",
            "Electromagnetic energy and momentum",
            "Gauge transformations and Lorenz gauge",
            "Green's functions for wave equations",
            "Scattering and diffraction of EM waves",
            "Magnetohydrodynamics",
            "Plasma physics and collective phenomena",
        ],
        PhysicsTopic.QUANTUM_MECHANICS: [
            "Time-independent perturbation theory",
            "Time-dependent perturbation theory and Fermi's golden rule",
            "Variational methods",
            "WKB approximation",
            "Scattering theory and partial waves",
            "Born approximation",
            "Identical particles and exchange symmetry",
            "Second quantization",
            "Angular momentum and Clebsch-Gordan coefficients",
            "Spin-orbit coupling and fine structure",
            "Hydrogen atom and Stark effect",
            "Density matrix and mixed states",
            "Quantum entanglement and Bell inequalities",
            "Path integral formulation",
            "Adiabatic theorem and Berry phase",
        ],
        PhysicsTopic.STATISTICAL_MECHANICS: [
            "Microcanonical, canonical, and grand canonical ensembles",
            "Partition functions and thermodynamic potentials",
            "Quantum statistical mechanics",
            "Bose-Einstein and Fermi-Dirac statistics",
            "Ideal quantum gases",
            "Phase transitions and critical phenomena",
            "Mean field theory and Landau theory",
            "Renormalization group basics",
            "Fluctuation-dissipation theorem",
            "Linear response theory",
            "Kinetic theory and Boltzmann equation",
            "Ising model and exact solutions",
            "Monte Carlo methods in statistical physics",
        ],
        PhysicsTopic.THERMODYNAMICS: [
            "Thermodynamic potentials and Maxwell relations",
            "Stability conditions and Le Chatelier's principle",
            "Phase equilibria and Clausius-Clapeyron equation",
            "Chemical potential and multicomponent systems",
            "Irreversible thermodynamics",
            "Onsager reciprocal relations",
            "Thermodynamics of black holes",
            "Fluctuations and the second law",
            "Third law and approaching absolute zero",
            "Non-equilibrium thermodynamics",
        ],
        PhysicsTopic.SPECIAL_RELATIVITY: [
            "Lorentz transformations and four-vectors",
            "Relativistic kinematics and dynamics",
            "Energy-momentum four-vector",
            "Relativistic collisions and threshold energies",
            "Covariant formulation of electromagnetism",
            "Thomas precession",
            "Relativistic Doppler effect",
            "Proper time and spacetime intervals",
            "Minkowski diagrams and causality",
            "Relativistic angular momentum",
        ],
        PhysicsTopic.GENERAL_RELATIVITY: [
            "Tensor calculus and differential geometry",
            "Einstein field equations",
            "Schwarzschild solution and black holes",
            "Geodesics and gravitational lensing",
            "Gravitational waves",
            "Kerr metric and rotating black holes",
            "Cosmological solutions (FLRW metric)",
            "Penrose diagrams",
            "Hawking radiation (semiclassical)",
            "Tests of general relativity",
        ],
        PhysicsTopic.CONDENSED_MATTER: [
            "Band theory and Bloch's theorem",
            "Tight-binding approximation",
            "Phonons and lattice dynamics",
            "Electron-phonon interactions",
            "BCS theory of superconductivity",
            "Ginzburg-Landau theory",
            "Magnetic ordering and spin waves",
            "Quantum Hall effect",
            "Topological insulators",
            "Fermi liquid theory",
            "Hubbard model",
            "Anderson localization",
        ],
        PhysicsTopic.NUCLEAR_PHYSICS: [
            "Nuclear shell model",
            "Liquid drop model and semi-empirical mass formula",
            "Nuclear reactions and cross-sections",
            "Alpha, beta, and gamma decay",
            "Nuclear fission and fusion",
            "Nucleon-nucleon scattering",
            "Isospin symmetry",
            "Nuclear magnetic resonance",
            "Collective nuclear excitations",
            "Neutron physics and nuclear reactors",
        ],
        PhysicsTopic.PARTICLE_PHYSICS: [
            "Dirac equation and spinors",
            "Quantum electrodynamics (QED) basics",
            "Feynman diagrams and cross-sections",
            "Gauge theories and symmetry breaking",
            "Electroweak unification",
            "Quantum chromodynamics (QCD)",
            "Quark model and hadron spectroscopy",
            "CP violation",
            "Neutrino oscillations",
            "Standard Model phenomenology",
        ],
        PhysicsTopic.OPTICS: [
            "Coherence and interference",
            "Diffraction theory (Fresnel and Fraunhofer)",
            "Polarization and Jones calculus",
            "Gaussian beam optics",
            "Nonlinear optics",
            "Laser physics and rate equations",
            "Optical cavities and resonators",
            "Fourier optics and spatial filtering",
            "Quantum optics basics",
            "Optical waveguides and fibers",
        ],
        PhysicsTopic.FLUID_MECHANICS: [
            "Navier-Stokes equations",
            "Potential flow and complex analysis",
            "Viscous flow and boundary layers",
            "Turbulence and Reynolds number",
            "Vortex dynamics",
            "Compressible flow and shock waves",
            "Instabilities (Rayleigh-Taylor, Kelvin-Helmholtz)",
            "Stokes flow and low Reynolds number",
            "Rotating fluids and geophysical flows",
            "Magnetohydrodynamic flows",
        ],
    }

    # Difficulty modifiers that make questions more graduate-level
    DIFFICULTY_MODIFIERS: List[str] = [
        "involving perturbative corrections",
        "requiring asymptotic analysis",
        "with non-trivial boundary conditions",
        "in a non-equilibrium context",
        "with coupled degrees of freedom",
        "requiring numerical estimation",
        "involving symmetry breaking",
        "with competing interactions",
        "near a critical point",
        "in the high-energy limit",
        "in the low-temperature limit",
        "with quantum corrections",
        "in curved spacetime",
        "with dissipation effects",
        "involving renormalization",
        "with topological considerations",
        "requiring Green's function methods",
        "in a many-body context",
        "with relativistic corrections",
        "involving spontaneous symmetry breaking",
    ]

    # Additional context that can make questions more specific
    ADDITIONAL_CONTEXTS: List[str] = [
        "Consider all relevant approximations and state their validity",
        "Derive the result from first principles",
        "Analyze the limiting cases",
        "Discuss the physical interpretation of each term",
        "Compare with experimental observations where applicable",
        "Include first-order corrections",
        "Consider finite-size effects",
        "Account for thermal fluctuations",
        "Analyze stability of the solution",
        "Discuss observable consequences",
    ]

    def __init__(
        self,
        topics: Optional[List[PhysicsTopic]] = None,
        weights: Optional[Dict[PhysicsTopic, float]] = None,
    ):
        """
        Initialize the topic sampler.

        Args:
            topics: List of topics to sample from (default: all topics)
            weights: Optional weights for each topic (default: uniform)
        """
        self.topics = topics or list(PhysicsTopic)
        self.weights = weights

        # Track what we've generated for diversity
        self._generated_subtopics: Dict[PhysicsTopic, List[str]] = {t: [] for t in self.topics}

    def sample(self, prefer_diverse: bool = True) -> TopicContext:
        """
        Sample a topic context for question generation.

        Args:
            prefer_diverse: If True, prefer subtopics we haven't used recently

        Returns:
            TopicContext with topic, subtopic, and difficulty modifier
        """
        # Sample topic
        if self.weights:
            weights = [self.weights.get(t, 1.0) for t in self.topics]
            topic = random.choices(self.topics, weights=weights, k=1)[0]
        else:
            topic = random.choice(self.topics)

        # Sample subtopic (prefer diverse if requested)
        available_subtopics = self.SUBTOPICS.get(topic, [])
        if prefer_diverse and self._generated_subtopics[topic]:
            # Prefer subtopics we haven't used
            unused = [s for s in available_subtopics if s not in self._generated_subtopics[topic]]
            if unused:
                subtopic = random.choice(unused)
            else:
                # Reset and start over
                self._generated_subtopics[topic] = []
                subtopic = random.choice(available_subtopics)
        else:
            subtopic = random.choice(available_subtopics) if available_subtopics else topic.value

        # Track for diversity
        self._generated_subtopics[topic].append(subtopic)

        # Sample difficulty modifier
        difficulty_modifier = random.choice(self.DIFFICULTY_MODIFIERS)

        # Optionally add additional context
        additional_context = (
            random.choice(self.ADDITIONAL_CONTEXTS) if random.random() < 0.5 else None
        )

        return TopicContext(
            topic=topic,
            subtopic=subtopic,
            difficulty_modifier=difficulty_modifier,
            additional_context=additional_context,
        )

    def sample_batch(self, n: int, ensure_topic_coverage: bool = True) -> List[TopicContext]:
        """
        Sample multiple topic contexts.

        Args:
            n: Number of contexts to sample
            ensure_topic_coverage: If True, try to cover all topics at least once

        Returns:
            List of TopicContext objects
        """
        contexts = []

        if ensure_topic_coverage and n >= len(self.topics):
            # First ensure each topic is covered
            for topic in self.topics:
                context = self.sample(prefer_diverse=True)
                # Override with specific topic
                subtopic = random.choice(self.SUBTOPICS.get(topic, [topic.value]))
                contexts.append(
                    TopicContext(
                        topic=topic,
                        subtopic=subtopic,
                        difficulty_modifier=random.choice(self.DIFFICULTY_MODIFIERS),
                        additional_context=(
                            random.choice(self.ADDITIONAL_CONTEXTS)
                            if random.random() < 0.5
                            else None
                        ),
                    )
                )

            # Fill remaining with random samples
            while len(contexts) < n:
                contexts.append(self.sample(prefer_diverse=True))
        else:
            # Just sample randomly
            for _ in range(n):
                contexts.append(self.sample(prefer_diverse=True))

        return contexts

    def get_coverage_stats(self) -> Dict[str, any]:
        """Get statistics about topic coverage."""
        return {
            "topics_used": {
                t.value: len(subtopics) for t, subtopics in self._generated_subtopics.items()
            },
            "total_generated": sum(len(s) for s in self._generated_subtopics.values()),
            "unique_subtopics": sum(
                len(set(s)) for s in self._generated_subtopics.values()
            ),
        }

    def reset_diversity_tracking(self):
        """Reset the diversity tracking."""
        self._generated_subtopics = {t: [] for t in self.topics}
