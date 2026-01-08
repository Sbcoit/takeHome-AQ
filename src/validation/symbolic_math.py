"""Symbolic math verification using SymPy.

This module provides PROGRAMMATIC verification of physics answers:
1. Parse equations from LaTeX/text
2. Verify dimensional consistency
3. Check algebraic simplifications
4. Validate limiting behavior symbolically

Unlike AI-based checks, this catches errors that models systematically miss.
"""

import logging
import re
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import sympy - it's optional but highly recommended
try:
    import sympy as sp
    from sympy import (
        symbols, Symbol, sympify, simplify, limit, oo,
        sin, cos, tan, exp, log, sqrt, pi, I, E,
        asin, acos, atan, sinh, cosh, tanh,  # Inverse trig and hyperbolic
        Abs, factorial, gamma, erf, besselj,
        diff, integrate, solve, series,
        Rational, Float, Integer,
        latex
    )
    from sympy.parsing.latex import parse_latex
    from sympy.physics.units import (
        Quantity, Dimension,
        length, mass, time, current, temperature,
        energy, force, velocity, acceleration, momentum,
        joule, meter, second, kilogram, ampere, kelvin,
        newton, watt, volt, ohm, coulomb, farad, henry,
        hertz, pascal, tesla, weber,
        c, hbar, G,
    )
    # Note: electron charge (e), m_e, m_p not available in all SymPy versions
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, convert_xor
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("SymPy not available - symbolic math checks will be skipped")


@dataclass
class DimensionalUnit:
    """Represents a physical dimension as powers of base units.

    Uses float exponents to handle sqrt and fractional powers correctly.
    For comparison, uses tolerance to handle floating point errors.
    """
    length: float = 0      # [L]
    mass: float = 0        # [M]
    time: float = 0        # [T]
    current: float = 0     # [I]
    temperature: float = 0 # [Θ]
    amount: float = 0      # [N]
    luminosity: float = 0  # [J]

    def __str__(self) -> str:
        parts = []
        # Format exponents nicely - show integers as integers
        def fmt_exp(val, letter):
            if abs(val) < 1e-10:
                return None
            if abs(val - 1) < 1e-10:
                return letter
            if abs(val - round(val)) < 1e-10:
                return f"{letter}^{int(round(val))}"
            return f"{letter}^{val:.2g}"

        for val, letter in [(self.mass, "M"), (self.length, "L"), (self.time, "T"),
                            (self.current, "I"), (self.temperature, "Θ")]:
            formatted = fmt_exp(val, letter)
            if formatted:
                parts.append(formatted)
        return " ".join(parts) if parts else "dimensionless"

    def __eq__(self, other) -> bool:
        if not isinstance(other, DimensionalUnit):
            return False
        # Use tolerance for floating point comparison
        tol = 1e-10
        return (abs(self.length - other.length) < tol and
                abs(self.mass - other.mass) < tol and
                abs(self.time - other.time) < tol and
                abs(self.current - other.current) < tol and
                abs(self.temperature - other.temperature) < tol)


# Common physics dimensions
DIMENSION_ENERGY = DimensionalUnit(mass=1, length=2, time=-2)      # [M L² T⁻²]
DIMENSION_MOMENTUM = DimensionalUnit(mass=1, length=1, time=-1)    # [M L T⁻¹]
DIMENSION_FORCE = DimensionalUnit(mass=1, length=1, time=-2)       # [M L T⁻²]
DIMENSION_VELOCITY = DimensionalUnit(length=1, time=-1)            # [L T⁻¹]
DIMENSION_ACCELERATION = DimensionalUnit(length=1, time=-2)        # [L T⁻²]
DIMENSION_LENGTH = DimensionalUnit(length=1)                       # [L]
DIMENSION_TIME = DimensionalUnit(time=1)                           # [T]
DIMENSION_MASS = DimensionalUnit(mass=1)                           # [M]
DIMENSION_FREQUENCY = DimensionalUnit(time=-1)                     # [T⁻¹]
DIMENSION_ANGULAR_MOMENTUM = DimensionalUnit(mass=1, length=2, time=-1)  # [M L² T⁻¹]
DIMENSION_ACTION = DIMENSION_ANGULAR_MOMENTUM                      # [M L² T⁻¹] (ħ units)
DIMENSION_ELECTRIC_FIELD = DimensionalUnit(mass=1, length=1, time=-3, current=-1)
DIMENSION_MAGNETIC_FIELD = DimensionalUnit(mass=1, time=-2, current=-1)
DIMENSION_CHARGE = DimensionalUnit(current=1, time=1)              # [I T]
DIMENSION_POTENTIAL = DimensionalUnit(mass=1, length=2, time=-3, current=-1)  # Voltage
DIMENSION_DIMENSIONLESS = DimensionalUnit()

# Map common physics quantities to their dimensions
QUANTITY_DIMENSIONS = {
    # Mechanics
    "energy": DIMENSION_ENERGY,
    "kinetic energy": DIMENSION_ENERGY,
    "potential energy": DIMENSION_ENERGY,
    "work": DIMENSION_ENERGY,
    "momentum": DIMENSION_MOMENTUM,
    "angular momentum": DIMENSION_ANGULAR_MOMENTUM,
    "force": DIMENSION_FORCE,
    "torque": DimensionalUnit(mass=1, length=2, time=-2),  # Same as energy but different context
    "velocity": DIMENSION_VELOCITY,
    "speed": DIMENSION_VELOCITY,
    "acceleration": DIMENSION_ACCELERATION,
    "length": DIMENSION_LENGTH,
    "distance": DIMENSION_LENGTH,
    "displacement": DIMENSION_LENGTH,
    "radius": DIMENSION_LENGTH,
    "wavelength": DIMENSION_LENGTH,
    "time": DIMENSION_TIME,
    "period": DIMENSION_TIME,
    "mass": DIMENSION_MASS,
    "frequency": DIMENSION_FREQUENCY,
    "angular frequency": DIMENSION_FREQUENCY,

    # Quantum mechanics
    "wave function": DimensionalUnit(length=-3, mass=0, time=0),  # [L^-3/2] for 3D
    "probability": DIMENSION_DIMENSIONLESS,
    "probability density": DimensionalUnit(length=-3),
    "action": DIMENSION_ACTION,

    # Electromagnetism
    "electric field": DIMENSION_ELECTRIC_FIELD,
    "magnetic field": DIMENSION_MAGNETIC_FIELD,
    "charge": DIMENSION_CHARGE,
    "current": DimensionalUnit(current=1),
    "voltage": DIMENSION_POTENTIAL,
    "potential": DIMENSION_POTENTIAL,
    "capacitance": DimensionalUnit(mass=-1, length=-2, time=4, current=2),
    "inductance": DimensionalUnit(mass=1, length=2, time=-2, current=-2),
    "resistance": DimensionalUnit(mass=1, length=2, time=-3, current=-2),

    # Thermodynamics
    "temperature": DimensionalUnit(temperature=1),
    "entropy": DimensionalUnit(mass=1, length=2, time=-2, temperature=-1),
    "heat capacity": DimensionalUnit(mass=1, length=2, time=-2, temperature=-1),
    "pressure": DimensionalUnit(mass=1, length=-1, time=-2),

    # Relativity
    "proper time": DIMENSION_TIME,
    "spacetime interval": DimensionalUnit(length=2),  # ds² has dimensions of length²

    # Dimensionless quantities
    "angle": DIMENSION_DIMENSIONLESS,
    "ratio": DIMENSION_DIMENSIONLESS,
    "coefficient": DIMENSION_DIMENSIONLESS,
    "index": DIMENSION_DIMENSIONLESS,
    "number": DIMENSION_DIMENSIONLESS,
}

# Map common physics SYMBOLS to their dimensions
# This allows us to compute dimensions of expressions like "G*M*m/r**2"
SYMBOL_DIMENSIONS = {
    # Fundamental constants
    "G": DimensionalUnit(mass=-1, length=3, time=-2),      # Gravitational constant [M⁻¹ L³ T⁻²]
    "c": DimensionalUnit(length=1, time=-1),               # Speed of light [L T⁻¹]
    "hbar": DimensionalUnit(mass=1, length=2, time=-1),    # Reduced Planck [M L² T⁻¹]
    "h": DimensionalUnit(mass=1, length=2, time=-1),       # Planck constant [M L² T⁻¹]
    "k_B": DimensionalUnit(mass=1, length=2, time=-2, temperature=-1),  # Boltzmann [M L² T⁻² Θ⁻¹]
    "epsilon_0": DimensionalUnit(mass=-1, length=-3, time=4, current=2),  # Permittivity
    "mu_0": DimensionalUnit(mass=1, length=1, time=-2, current=-2),       # Permeability
    "e": DimensionalUnit(current=1, time=1),               # Elementary charge [I T]

    # Common variable conventions (context-dependent, but common defaults)
    "m": DimensionalUnit(mass=1),                          # mass
    "M": DimensionalUnit(mass=1),                          # mass (often central/total mass)
    "r": DimensionalUnit(length=1),                        # radius/distance
    "R": DimensionalUnit(length=1),                        # radius
    "x": DimensionalUnit(length=1),                        # position
    "y": DimensionalUnit(length=1),                        # position
    "z": DimensionalUnit(length=1),                        # position
    "t": DimensionalUnit(time=1),                          # time
    "v": DimensionalUnit(length=1, time=-1),               # velocity
    "a": DimensionalUnit(length=1, time=-2),               # acceleration
    "F": DimensionalUnit(mass=1, length=1, time=-2),       # force
    "p": DimensionalUnit(mass=1, length=1, time=-1),       # momentum
    "E": DimensionalUnit(mass=1, length=2, time=-2),       # energy
    "U": DimensionalUnit(mass=1, length=2, time=-2),       # potential energy
    "T": DimensionalUnit(mass=1, length=2, time=-2),       # kinetic energy (or temperature - context)
    "V": DimensionalUnit(mass=1, length=2, time=-2),       # potential (energy context)
    "L": DimensionalUnit(mass=1, length=2, time=-1),       # angular momentum
    "J": DimensionalUnit(mass=1, length=2, time=-1),       # angular momentum
    "omega": DimensionalUnit(time=-1),                     # angular frequency
    "tau": DimensionalUnit(time=1),                        # time constant / proper time
    "lambda_": DimensionalUnit(length=1),                  # wavelength
    "k": DimensionalUnit(length=-1),                       # wave number
    "q": DimensionalUnit(current=1, time=1),               # charge
    "Q": DimensionalUnit(current=1, time=1),               # charge
    "I": DimensionalUnit(current=1),                       # current
    "B": DimensionalUnit(mass=1, time=-2, current=-1),     # magnetic field

    # Dimensionless
    "pi": DimensionalUnit(),
    "theta": DimensionalUnit(),
    "phi": DimensionalUnit(),
    "alpha": DimensionalUnit(),                            # fine structure constant (usually)
    "n": DimensionalUnit(),                                # quantum number / index
    "N": DimensionalUnit(),                                # count
}


def compute_expression_dimensions(expr, symbol_dims: dict = None) -> Optional[DimensionalUnit]:
    """
    Compute the dimensions of a SymPy expression by tracking dimensions through operations.

    Args:
        expr: A SymPy expression
        symbol_dims: Optional dict mapping symbol names to DimensionalUnit

    Returns:
        DimensionalUnit if computable, None if dimensions are inconsistent or unknown
    """
    if not SYMPY_AVAILABLE:
        return None

    if symbol_dims is None:
        symbol_dims = SYMBOL_DIMENSIONS

    try:
        return _compute_dims_recursive(expr, symbol_dims)
    except Exception as e:
        logger.debug(f"Could not compute dimensions: {e}")
        return None


def _compute_dims_recursive(expr, symbol_dims: dict) -> Optional[DimensionalUnit]:
    """Recursively compute dimensions of an expression."""
    # Handle numbers - dimensionless
    if expr.is_number:
        return DimensionalUnit()

    # Handle symbols
    if expr.is_Symbol:
        sym_name = str(expr)
        # Try exact match first
        if sym_name in symbol_dims:
            return symbol_dims[sym_name]
        # Try without subscript (e.g., "m_0" -> "m")
        base_name = sym_name.split('_')[0]
        if base_name in symbol_dims:
            return symbol_dims[base_name]
        # Unknown symbol - return None to indicate we can't determine dimensions
        logger.debug(f"Unknown symbol for dimensional analysis: {sym_name}")
        return None

    # Handle multiplication: dimensions multiply (add exponents)
    if expr.is_Mul:
        result = DimensionalUnit()
        for arg in expr.args:
            arg_dim = _compute_dims_recursive(arg, symbol_dims)
            if arg_dim is None:
                return None
            result = DimensionalUnit(
                length=result.length + arg_dim.length,
                mass=result.mass + arg_dim.mass,
                time=result.time + arg_dim.time,
                current=result.current + arg_dim.current,
                temperature=result.temperature + arg_dim.temperature,
            )
        return result

    # Handle division/power: a**n multiplies exponents by n
    if expr.is_Pow:
        base, exp = expr.as_base_exp()
        base_dim = _compute_dims_recursive(base, symbol_dims)
        if base_dim is None:
            return None
        # Exponent must be dimensionless and ideally a number
        if not exp.is_number:
            # Variable exponent - can only work if base is dimensionless
            if base_dim == DimensionalUnit():
                return DimensionalUnit()
            return None
        n = float(exp)
        # Compute fractional exponents - don't round, let float handle it
        return DimensionalUnit(
            length=base_dim.length * n,
            mass=base_dim.mass * n,
            time=base_dim.time * n,
            current=base_dim.current * n,
            temperature=base_dim.temperature * n,
        )

    # Handle addition/subtraction: all terms must have same dimensions
    if expr.is_Add:
        dims = []
        for arg in expr.args:
            arg_dim = _compute_dims_recursive(arg, symbol_dims)
            if arg_dim is None:
                return None
            dims.append(arg_dim)
        # Check all dimensions match
        if len(dims) > 0:
            first = dims[0]
            for d in dims[1:]:
                if d != first:
                    logger.warning(f"Dimensional mismatch in addition: {first} vs {d}")
                    return None  # Dimensional inconsistency!
            return first
        return DimensionalUnit()

    # Handle functions - most return dimensionless or preserve dimensions
    if expr.is_Function:
        func_name = expr.func.__name__
        # Trig functions require dimensionless argument, return dimensionless
        if func_name in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']:
            arg_dim = _compute_dims_recursive(expr.args[0], symbol_dims)
            if arg_dim is not None and arg_dim != DimensionalUnit():
                logger.warning(f"Trig function {func_name} has non-dimensionless argument")
            return DimensionalUnit()
        # exp, log require dimensionless
        if func_name in ['exp', 'log', 'ln']:
            return DimensionalUnit()
        # sqrt preserves dimensions with 1/2 power
        if func_name == 'sqrt':
            arg_dim = _compute_dims_recursive(expr.args[0], symbol_dims)
            if arg_dim is None:
                return None
            return DimensionalUnit(
                length=arg_dim.length / 2,
                mass=arg_dim.mass / 2,
                time=arg_dim.time / 2,
                current=arg_dim.current / 2,
                temperature=arg_dim.temperature / 2,
            )

    # Default: can't determine
    logger.debug(f"Cannot determine dimensions for expression type: {type(expr)}")
    return None


class SymbolicMathValidator:
    """
    Validates physics answers using symbolic mathematics.

    This provides PROGRAMMATIC verification that catches errors
    AI models systematically miss:
    - Dimensional analysis via symbol tracking
    - Algebraic verification
    - Limiting case checking
    - Expression equivalence
    """

    def __init__(self):
        """Initialize the symbolic math validator."""
        if not SYMPY_AVAILABLE:
            logger.warning("SymPy not available - validation will be limited")

        # Common physics symbols
        if SYMPY_AVAILABLE:
            self._setup_physics_symbols()

    def _setup_physics_symbols(self):
        """Set up common physics symbols with assumed properties."""
        # Real positive quantities
        self.m, self.M = symbols('m M', real=True, positive=True)  # mass
        self.r, self.R = symbols('r R', real=True, positive=True)  # radius
        self.v = symbols('v', real=True)  # velocity (can be negative)
        self.t = symbols('t', real=True, nonnegative=True)  # time
        self.E = symbols('E', real=True)  # energy
        self.omega, self.w = symbols('omega w', real=True)  # angular frequency
        self.k = symbols('k', real=True, positive=True)  # wave number
        self.n = symbols('n', integer=True, nonnegative=True)  # quantum number
        self.l = symbols('l', integer=True, nonnegative=True)  # angular quantum number
        self.hbar = symbols('hbar', real=True, positive=True)  # reduced Planck
        self.c_sym = symbols('c', real=True, positive=True)  # speed of light
        self.G_sym = symbols('G', real=True, positive=True)  # gravitational constant
        self.e_sym = symbols('e', real=True, positive=True)  # electron charge
        self.epsilon_0 = symbols('epsilon_0', real=True, positive=True)
        self.mu_0 = symbols('mu_0', real=True, positive=True)

    def validate(
        self,
        query: str,
        answer: str,
        reasoning: str,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Validate a physics answer using symbolic math.

        Args:
            query: The physics question
            answer: The final answer expression
            reasoning: The derivation/reasoning

        Returns:
            Tuple of (passed, details, feedback)
        """
        if not SYMPY_AVAILABLE:
            return True, {"skipped": True, "reason": "SymPy not available"}, ""

        details = {
            "expression_parsed": False,
            "dimensional_check": None,
            "algebraic_check": None,
            "limiting_cases": [],
            "numerical_sanity": None,
        }

        try:
            # Step 1: Try to parse the answer expression
            expr = self._parse_physics_expression(answer)
            if expr is not None:
                details["expression_parsed"] = True
                details["parsed_expression"] = str(expr)

                # Step 2: Check for common algebraic issues
                alg_check = self._check_algebraic_issues(expr)
                details["algebraic_check"] = alg_check

                # Step 3: Try to extract and check limiting cases
                limits = self._check_limiting_cases(expr, query)
                details["limiting_cases"] = limits

                # Step 4: Numerical sanity (if expression has numerical values)
                num_check = self._check_numerical_sanity(expr)
                details["numerical_sanity"] = num_check

                # Step 5: PROGRAMMATIC dimensional analysis (NEW)
                # Compute actual dimensions from the expression symbols
                prog_dim_check = self._check_programmatic_dimensions(expr, query)
                details["programmatic_dimensional_check"] = prog_dim_check
            else:
                details["parse_error"] = "Could not parse expression"

            # Step 6: Dimensional analysis from text (heuristic)
            dim_check = self._infer_dimensional_check(query, answer)
            details["dimensional_check"] = dim_check

            # Determine if passed
            passed = self._evaluate_results(details)
            feedback = self._generate_feedback(details) if not passed else ""

            return passed, details, feedback

        except Exception as e:
            logger.error(f"Symbolic validation error: {e}")
            # Return failure with error details - don't silently pass
            return False, {"error": str(e), "expression_parsed": False}, f"SYMBOLIC MATH ERROR: {e}"

    def _parse_physics_expression(self, expr_str: str) -> Optional[Any]:
        r"""
        Try to parse a physics expression into SymPy.

        Handles common physics notation:
        - LaTeX: \frac{}, \sqrt{}, \hbar, etc.
        - Plain text: E = mc^2, hbar, etc.
        """
        if not SYMPY_AVAILABLE:
            return None

        # Clean up the expression
        expr_str = expr_str.strip()

        # Remove markdown bold markers early (before splitting on =)
        expr_str = re.sub(r'^\*\*(.+?)\*\*$', r'\1', expr_str)
        expr_str = expr_str.strip()

        # Remove equation formatting if present
        if "=" in expr_str:
            # Take the right-hand side
            parts = expr_str.split("=")
            expr_str = parts[-1].strip()

        # Try LaTeX parsing first
        try:
            if "\\" in expr_str or "{" in expr_str:
                # Looks like LaTeX
                expr = parse_latex(expr_str)
                return expr
        except Exception:
            pass

        # Try standard parsing with physics transformations
        try:
            # Replace common physics notation
            expr_str = self._normalize_physics_notation(expr_str)

            transformations = standard_transformations + (
                implicit_multiplication_application,
                convert_xor,
            )

            # Create symbols for Greek letters
            alpha, beta, gamma, delta, epsilon = symbols('alpha beta gamma delta epsilon', real=True)
            lambda_, Lambda = symbols('lambda_ Lambda', real=True, positive=True)
            mu, nu = symbols('mu nu', real=True)
            phi, Phi, psi, Psi = symbols('phi Phi psi Psi', real=True)
            theta, Theta = symbols('theta Theta', real=True)
            sigma, tau, rho = symbols('sigma tau rho', real=True)
            kappa, chi, zeta, eta, xi = symbols('kappa chi zeta eta xi', real=True)
            Omega = symbols('Omega', real=True)
            k_B = symbols('k_B', real=True, positive=True)
            T = symbols('T', real=True, positive=True)  # Temperature
            N = symbols('N', integer=True, positive=True)  # Number
            g = symbols('g', real=True)  # Coupling constant or gravity
            a = symbols('a', real=True)
            L = symbols('L', real=True, positive=True)
            q = symbols('q', real=True)
            A, K, J = symbols('A K J', real=True)

            local_dict = {
                'hbar': self.hbar,
                'c': self.c_sym,
                'G': self.G_sym,
                'e': self.e_sym,
                'pi': pi,
                'i': I,
                'E': self.E,
                'm': self.m,
                'M': self.M,
                'r': self.r,
                'R': self.R,
                'v': self.v,
                't': self.t,
                'omega': self.omega,
                'w': self.w,
                'k': self.k,
                'n': self.n,
                'l': self.l,
                'epsilon_0': self.epsilon_0,
                'mu_0': self.mu_0,
                # Greek letters
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma,
                'delta': delta,
                'epsilon': epsilon,
                'lambda_': lambda_,
                'Lambda': Lambda,
                'mu': mu,
                'nu': nu,
                'phi': phi,
                'Phi': Phi,
                'psi': psi,
                'Psi': Psi,
                'theta': theta,
                'Theta': Theta,
                'sigma': sigma,
                'tau': tau,
                'rho': rho,
                'kappa': kappa,
                'chi': chi,
                'zeta': zeta,
                'eta': eta,
                'xi': xi,
                'Omega': Omega,
                # Physics constants/variables
                'k_B': k_B,
                'T': T,
                'N': N,
                'g': g,
                'a': a,
                'L': L,
                'q': q,
                'A': A,
                'K': K,
                'J': J,
                # Math functions
                'sqrt': sqrt,
                'sin': sin,
                'cos': cos,
                'tan': tan,
                'exp': exp,
                'log': log,
                'oo': oo,
                # Inverse trig functions
                'asin': asin,
                'acos': acos,
                'atan': atan,
                'arcsin': asin,
                'arccos': acos,
                'arctan': atan,
                # Hyperbolic functions
                'sinh': sinh,
                'cosh': cosh,
                'tanh': tanh,
            }

            expr = parse_expr(expr_str, local_dict=local_dict, transformations=transformations)
            return expr
        except Exception as e:
            logger.debug(f"Could not parse expression '{expr_str}': {e}")
            return None

    def _normalize_physics_notation(self, s: str) -> str:
        """Convert common physics notation to SymPy-parseable form."""
        # Remove markdown bold markers
        s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
        s = s.strip()

        # Convert square brackets to parentheses (common in physics notation)
        s = s.replace('[', '(')
        s = s.replace(']', ')')

        # Handle nested fractions - process from innermost to outermost
        # Keep processing until no more \frac patterns remain
        max_iterations = 10  # Prevent infinite loops
        for _ in range(max_iterations):
            # Match \frac{...}{...} where the braces contain non-brace chars or already-converted content
            new_s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
            if new_s == s:
                break
            s = new_s

        # Handle \sqrt{...}
        for _ in range(max_iterations):
            new_s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
            if new_s == s:
                break
            s = new_s

        # Replace common LaTeX commands (order matters)
        # Use both single and double backslash versions to handle different input formats
        latex_commands = [
            # Greek letters (common in physics) - both \cmd and \\cmd formats
            ('\\hbar', 'hbar'),
            ('\\alpha', 'alpha'),
            ('\\beta', 'beta'),
            ('\\gamma', 'gamma'),
            ('\\delta', 'delta'),
            ('\\epsilon', 'epsilon'),
            ('\\lambda', 'lambda_'),  # lambda is Python keyword
            ('\\Lambda', 'Lambda'),
            ('\\mu', 'mu'),
            ('\\nu', 'nu'),
            ('\\omega', 'omega'),
            ('\\Omega', 'Omega'),
            ('\\phi', 'phi'),
            ('\\Phi', 'Phi'),
            ('\\psi', 'psi'),
            ('\\Psi', 'Psi'),
            ('\\theta', 'theta'),
            ('\\Theta', 'Theta'),
            ('\\sigma', 'sigma'),
            ('\\tau', 'tau'),
            ('\\rho', 'rho'),
            ('\\kappa', 'kappa'),
            ('\\chi', 'chi'),
            ('\\zeta', 'zeta'),
            ('\\eta', 'eta'),
            ('\\xi', 'xi'),
            # Math constants
            ('\\pi', 'pi'),
            ('\\infty', 'oo'),
            # Physical constants
            ('\\epsilon_0', 'epsilon_0'),
            ('\\mu_0', 'mu_0'),
            ('k_B', 'k_B'),
            # Operators
            ('\\cdot', '*'),
            ('\\times', '*'),
            ('\\left', ''),
            ('\\right', ''),
            ('\\bigg', ''),
            ('\\Big', ''),
            ('\\big', ''),
        ]

        for pattern, replacement in latex_commands:
            s = s.replace(pattern, replacement)

        # Handle subscripts: omega_0 -> omega_0 (keep as single symbol name)
        # Remove braces from subscripts: x_{0} -> x_0
        s = re.sub(r'_\{([^{}]+)\}', r'_\1', s)
        # Keep subscripts as part of the symbol name (don't convert omega_0 to omega0)
        # This prevents omega0 from being interpreted as omega*0

        # Handle superscripts: x^2 -> x**2, x^{2} -> x**2
        s = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', s)  # x^{2+n} -> x**(2+n)
        s = re.sub(r'\^([0-9]+)', r'**\1', s)  # x^2 -> x**2
        s = re.sub(r'\^([a-zA-Z])', r'**\1', s)  # x^n -> x**n

        # Remove remaining LaTeX artifacts
        s = s.replace('\\', '')  # Remove remaining backslashes
        s = s.replace('$', '')  # Remove dollar signs

        # Handle space as multiplication first (before other processing)
        s = re.sub(r'\s+', ' ', s)  # Normalize whitespace
        s = s.strip()

        # Handle arcsin, arccos, arctan BEFORE implicit multiplication
        # These might appear as "arc sin", "arc*sin", "arcsin", or with LaTeX \arcsin
        # Use unique placeholders to protect these function names from being split
        # (e.g., "asin" becoming "a*sin" due to implicit multiplication rules)
        s = s.replace('arcsin', '__ASIN__')
        s = s.replace('arccos', '__ACOS__')
        s = s.replace('arctan', '__ATAN__')
        s = re.sub(r'arc\s*\*?\s*sin', '__ASIN__', s)
        s = re.sub(r'arc\s*\*?\s*cos', '__ACOS__', s)
        s = re.sub(r'arc\s*\*?\s*tan', '__ATAN__', s)
        # Also protect asin/acos/atan if already written that way
        s = s.replace('asin', '__ASIN__')
        s = s.replace('acos', '__ACOS__')
        s = s.replace('atan', '__ATAN__')
        # Protect other multi-char functions too
        s = s.replace('sinh', '__SINH__')
        s = s.replace('cosh', '__COSH__')
        s = s.replace('tanh', '__TANH__')
        s = s.replace('sqrt', '__SQRT__')
        s = s.replace('log', '__LOG__')
        s = s.replace('exp', '__EXP__')
        s = s.replace('abs', '__ABS__')

        # Add implicit multiplication
        # )( -> )*(
        s = re.sub(r'\)\s*\(', ')*(', s)
        # )letter -> )*letter
        s = re.sub(r'\)\s*([a-zA-Z])', r')*\1', s)
        # number( -> number*(
        s = re.sub(r'(\d)\s*\(', r'\1*(', s)

        # letter( -> letter*(  (but not for functions like sqrt, sin, cos)
        s = re.sub(r'([a-zA-Z])(\()', r'\1*\2', s)
        # Then fix function calls by removing the * we just added
        for func in ['sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']:
            s = s.replace(f'{func}*(', f'{func}(')

        # But add * before function names if preceded by letter/number: "2gsqrt" -> "2*g*sqrt"
        for func in ['sqrt', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh']:
            s = re.sub(rf'([a-zA-Z0-9]){func}\(', rf'\1*{func}(', s)

        # Add * between number and letter: "2g" -> "2*g"
        s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)

        # Handle implicit multiplication with spaces: "3 lambda" -> "3*lambda"
        s = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', s)
        s = re.sub(r'([a-zA-Z_])\s+([a-zA-Z])', r'\1*\2', s)
        s = re.sub(r'([a-zA-Z_])\s+(\d)', r'\1*\2', s)
        s = re.sub(r'\)\s+([a-zA-Z\d])', r')*\1', s)

        # Clean up any remaining spaces (but preserve spaces around operators)
        # Replace " " with "*" only when between alphanumerics
        s = re.sub(r'([a-zA-Z0-9_])\s+([a-zA-Z0-9_])', r'\1*\2', s)
        s = re.sub(r'\s+', '', s)  # Remove remaining spaces

        # Fix operators that got messed up: *-* -> -, *+* -> +
        s = s.replace('*-*', '-')
        s = s.replace('*+*', '+')
        s = s.replace('*=*', '=')
        s = s.replace('-*', '-')
        s = s.replace('+*', '+')
        s = s.replace('*-', '-')
        s = s.replace('*+', '+')
        s = s.replace('(-', '(-')
        s = s.replace('(+', '(+')

        # Handle adjacent known symbols that got concatenated
        # e.g., "hbaromega" should be "hbar*omega"
        known_pairs = [
            ('hbaromega', 'hbar*omega'),
            ('hbarsigma', 'hbar*sigma'),
            ('hbardelta', 'hbar*delta'),
            ('hbarpi', 'hbar*pi'),
            ('omegasqrt', 'omega*sqrt'),
            ('gammahbar', 'gamma*hbar'),
            ('alphasqrt', 'alpha*sqrt'),
            ('betasqrt', 'beta*sqrt'),
        ]
        for old, new in known_pairs:
            s = s.replace(old, new)

        # More general: insert * between concatenated Greek letter names
        greek_letters = ['hbar', 'omega', 'Omega', 'alpha', 'beta', 'gamma', 'delta',
                        'epsilon', 'lambda_', 'Lambda', 'theta', 'Theta', 'phi', 'Phi',
                        'psi', 'Psi', 'sigma', 'tau', 'rho', 'kappa', 'chi', 'zeta',
                        'eta', 'xi', 'mu', 'nu', 'pi']
        for g1 in greek_letters:
            for g2 in greek_letters:
                s = s.replace(f'{g1}{g2}', f'{g1}*{g2}')

        # Unicode to ASCII
        unicode_replacements = [
            ('×', '*'),
            ('·', '*'),
            ('−', '-'),
            ('≈', '='),
            ('→', ''),
            ('∞', 'oo'),
            ('²', '**2'),
            ('³', '**3'),
            ('⁴', '**4'),
            ('₀', '0'),
            ('₁', '1'),
            ('₂', '2'),
            ('α', 'alpha'),
            ('β', 'beta'),
            ('γ', 'gamma'),
            ('δ', 'delta'),
            ('ε', 'epsilon'),
            ('λ', 'lambda_'),
            ('μ', 'mu'),
            ('ν', 'nu'),
            ('ω', 'omega'),
            ('Ω', 'Omega'),
            ('π', 'pi'),
            ('σ', 'sigma'),
            ('τ', 'tau'),
            ('φ', 'phi'),
            ('ψ', 'psi'),
            ('θ', 'theta'),
            ('ρ', 'rho'),
            ('ℏ', 'hbar'),
        ]

        for old, new in unicode_replacements:
            s = s.replace(old, new)

        # Restore protected function names from placeholders
        s = s.replace('__ASIN__', 'asin')
        s = s.replace('__ACOS__', 'acos')
        s = s.replace('__ATAN__', 'atan')
        s = s.replace('__SINH__', 'sinh')
        s = s.replace('__COSH__', 'cosh')
        s = s.replace('__TANH__', 'tanh')
        s = s.replace('__SQRT__', 'sqrt')
        s = s.replace('__LOG__', 'log')
        s = s.replace('__EXP__', 'exp')
        s = s.replace('__ABS__', 'abs')

        return s

    def _check_algebraic_issues(self, expr) -> Dict[str, Any]:
        """Check for common algebraic issues in the expression."""
        if not SYMPY_AVAILABLE:
            return {"skipped": True}

        issues = []

        try:
            # Check for division by zero potential
            # Get denominators
            if hasattr(expr, 'as_numer_denom'):
                numer, denom = expr.as_numer_denom()
                if denom != 1:
                    # Check if denominator can be zero
                    free_syms = denom.free_symbols
                    if free_syms:
                        issues.append({
                            "type": "potential_division_by_zero",
                            "denominator": str(denom),
                            "variables": [str(s) for s in free_syms],
                        })

            # Note: We no longer flag imaginary unit as an issue because:
            # 1. Physics commonly uses 'I' for current, moment of inertia, intensity
            # 2. SymPy's parser often misinterprets these as imaginary unit
            # 3. Quantum mechanics legitimately uses complex numbers
            # The sanity check (AI-based) is better at determining if imaginary is appropriate

            # Check for undefined at certain limits
            simplified = simplify(expr)

            return {
                "issues": issues,
                "simplified_form": str(simplified),
                "passed": len(issues) == 0 or all(
                    issue.get("type") == "potential_division_by_zero"
                    for issue in issues
                )  # Division warnings are soft failures
            }

        except Exception as e:
            return {"error": str(e), "passed": True}

    def _check_limiting_cases(self, expr, query: str) -> List[Dict[str, Any]]:
        """Check expression behavior at limiting cases."""
        if not SYMPY_AVAILABLE:
            return []

        results = []
        free_syms = expr.free_symbols

        # Common physics limits to check
        limit_checks = [
            # (symbol_name, limit_value, expected_behavior_keywords)
            ("v", 0, ["rest", "zero velocity", "stationary"]),
            ("v", self.c_sym, ["relativistic", "speed of light"]),
            ("r", 0, ["origin", "center", "r=0"]),
            ("r", sp.oo, ["infinity", "far away", "large r"]),
            ("t", 0, ["initial", "t=0", "start"]),
            ("t", sp.oo, ["long time", "equilibrium", "steady state"]),
            ("hbar", 0, ["classical limit", "classical"]),
            ("n", sp.oo, ["large n", "classical limit", "correspondence"]),
            ("omega", 0, ["static", "zero frequency"]),
            ("omega", sp.oo, ["high frequency"]),
        ]

        for sym_name, limit_val, keywords in limit_checks:
            # Check if this symbol is in the expression
            matching_sym = None
            for s in free_syms:
                if str(s) == sym_name or str(s).startswith(sym_name):
                    matching_sym = s
                    break

            if matching_sym is None:
                continue

            try:
                # Calculate the limit
                lim_result = limit(expr, matching_sym, limit_val)

                # Check if it's well-behaved
                is_finite = lim_result.is_finite
                is_zero = lim_result == 0
                is_infinite = lim_result.is_infinite

                results.append({
                    "variable": sym_name,
                    "limit": str(limit_val),
                    "result": str(lim_result),
                    "is_finite": is_finite,
                    "is_zero": is_zero,
                    "is_infinite": is_infinite,
                })
            except Exception as e:
                logger.debug(f"Could not compute limit {sym_name} -> {limit_val}: {e}")

        return results

    def _check_numerical_sanity(self, expr) -> Dict[str, Any]:
        """Check if numerical values in the expression are sane."""
        if not SYMPY_AVAILABLE:
            return {"skipped": True}

        try:
            # Extract numerical coefficients
            coefficients = []

            def extract_numbers(e):
                if e.is_number:
                    coefficients.append(float(e))
                for arg in e.args:
                    extract_numbers(arg)

            extract_numbers(expr)

            issues = []
            for coef in coefficients:
                # Check for suspiciously large/small numbers
                if abs(coef) > 1e30 and abs(coef) != float('inf'):
                    issues.append(f"Very large coefficient: {coef:.2e}")
                elif abs(coef) < 1e-30 and coef != 0:
                    issues.append(f"Very small coefficient: {coef:.2e}")

            return {
                "coefficients_found": len(coefficients),
                "issues": issues,
                "passed": len(issues) == 0
            }

        except Exception as e:
            return {"error": str(e), "passed": True}

    def _check_programmatic_dimensions(self, expr, query: str) -> Dict[str, Any]:
        """
        Programmatically check dimensions by computing them from the expression.

        This catches errors like "GMmL/c²" having dimensions [ML³T⁻¹] instead of [ML²T⁻¹].
        """
        if not SYMPY_AVAILABLE:
            return {"skipped": True}

        try:
            # Compute actual dimensions from the expression
            actual_dims = compute_expression_dimensions(expr)

            if actual_dims is None:
                return {
                    "status": "could_not_compute",
                    "note": "Could not determine dimensions (unknown symbols or complex expression)",
                    "passed": True  # Don't fail if we can't compute
                }

            # Try to infer expected dimensions from the question
            expected_dims = self._infer_expected_dimensions_from_query(query)

            if expected_dims is None:
                return {
                    "status": "computed_only",
                    "actual_dimensions": str(actual_dims),
                    "note": "Computed dimensions but could not determine expected dimensions from question",
                    "passed": True  # Don't fail if we can't determine expected
                }

            # Compare actual vs expected
            dims_match = actual_dims == expected_dims

            result = {
                "status": "checked",
                "actual_dimensions": str(actual_dims),
                "expected_dimensions": str(expected_dims),
                "match": dims_match,
                "passed": dims_match,
            }

            if not dims_match:
                result["error"] = (
                    f"DIMENSIONAL ERROR: Answer has dimensions [{actual_dims}] "
                    f"but should have [{expected_dims}]"
                )
                logger.warning(
                    f"Dimensional mismatch detected: actual={actual_dims}, expected={expected_dims}"
                )

            return result

        except Exception as e:
            logger.debug(f"Programmatic dimensional check failed: {e}")
            return {"error": str(e), "passed": True}

    def _infer_expected_dimensions_from_query(self, query: str) -> Optional[DimensionalUnit]:
        """
        Infer expected dimensions from keywords in the question.

        Returns DimensionalUnit if we can determine expected dimensions, None otherwise.
        """
        query_lower = query.lower()

        # Direct quantity matches
        for quantity, dimension in QUANTITY_DIMENSIONS.items():
            # Look for phrases like "find the energy", "calculate the momentum", etc.
            patterns = [
                f"find the {quantity}",
                f"calculate the {quantity}",
                f"compute the {quantity}",
                f"determine the {quantity}",
                f"what is the {quantity}",
                f"derive the {quantity}",
                f"obtain the {quantity}",
                f"expression for the {quantity}",
                f"formula for the {quantity}",
            ]
            for pattern in patterns:
                if pattern in query_lower:
                    return dimension

        # Keyword-based inference
        keyword_dims = [
            (["angular momentum", "spin", "orbital angular momentum"], DIMENSION_ANGULAR_MOMENTUM),
            (["energy", "work", "hamiltonian", "lagrangian", "kinetic energy", "potential energy"], DIMENSION_ENERGY),
            (["momentum", "impulse", "linear momentum"], DIMENSION_MOMENTUM),
            (["force", "tension", "normal force", "friction"], DIMENSION_FORCE),
            (["velocity", "speed"], DIMENSION_VELOCITY),
            (["acceleration"], DIMENSION_ACCELERATION),
            (["length", "distance", "radius", "height", "wavelength", "displacement", "amplitude"], DIMENSION_LENGTH),
            (["time", "period", "lifetime", "decay time"], DIMENSION_TIME),
            (["frequency", "angular frequency"], DIMENSION_FREQUENCY),
            (["probability", "probability amplitude", "transmission coefficient"], DIMENSION_DIMENSIONLESS),
            (["electric field", "field strength"], DIMENSION_ELECTRIC_FIELD),
            (["magnetic field", "b-field"], DIMENSION_MAGNETIC_FIELD),
            (["charge", "total charge"], DIMENSION_CHARGE),
            (["voltage", "potential difference", "emf", "electric potential"], DIMENSION_POTENTIAL),
            (["mass"], DIMENSION_MASS),
            (["pressure", "stress"], DimensionalUnit(mass=1, length=-1, time=-2)),
            (["power", "luminosity"], DimensionalUnit(mass=1, length=2, time=-3)),
            (["area", "cross section"], DimensionalUnit(length=2)),
            (["volume"], DimensionalUnit(length=3)),
            (["density", "mass density"], DimensionalUnit(mass=1, length=-3)),
            (["torque", "moment"], DimensionalUnit(mass=1, length=2, time=-2)),
            (["action"], DIMENSION_ACTION),
        ]

        for keywords, dim in keyword_dims:
            for kw in keywords:
                if kw in query_lower:
                    return dim

        return None

    def _infer_dimensional_check(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Infer expected dimensions from the question and check answer.

        This is a heuristic check based on keywords in the question.
        """
        query_lower = query.lower()

        # Try to identify what quantity is being asked for
        expected_dim = None
        quantity_name = None

        for quantity, dimension in QUANTITY_DIMENSIONS.items():
            if quantity in query_lower:
                expected_dim = dimension
                quantity_name = quantity
                break

        # Also check for specific keywords
        keyword_dims = [
            (["energy", "work", "hamiltonian"], DIMENSION_ENERGY),
            (["momentum", "impulse"], DIMENSION_MOMENTUM),
            (["force"], DIMENSION_FORCE),
            (["velocity", "speed"], DIMENSION_VELOCITY),
            (["acceleration"], DIMENSION_ACCELERATION),
            (["length", "distance", "radius", "height", "wavelength"], DIMENSION_LENGTH),
            (["time", "period"], DIMENSION_TIME),
            (["frequency", "angular frequency"], DIMENSION_FREQUENCY),
            (["probability", "probability amplitude"], DIMENSION_DIMENSIONLESS),
            (["electric field"], DIMENSION_ELECTRIC_FIELD),
            (["magnetic field"], DIMENSION_MAGNETIC_FIELD),
            (["charge"], DIMENSION_CHARGE),
            (["voltage", "potential difference", "emf"], DIMENSION_POTENTIAL),
        ]

        for keywords, dim in keyword_dims:
            for kw in keywords:
                if kw in query_lower and expected_dim is None:
                    expected_dim = dim
                    quantity_name = kw
                    break

        if expected_dim is None:
            return {
                "status": "could_not_infer",
                "note": "Could not determine expected dimensions from question"
            }

        # Try to identify dimensions in the answer
        # This is simplified - look for common unit patterns
        answer_lower = answer.lower()

        return {
            "status": "inferred",
            "expected_quantity": quantity_name,
            "expected_dimensions": str(expected_dim),
            "note": "Dimensional check is based on keyword inference - verify manually"
        }

    def _evaluate_results(self, details: Dict[str, Any]) -> bool:
        """Evaluate all check results to determine pass/fail."""
        # If we couldn't parse the expression, FAIL - expressions must be parseable
        # This ensures answers use valid mathematical notation that can be verified
        if not details.get("expression_parsed", False):
            return False

        # Check algebraic issues
        alg = details.get("algebraic_check") or {}
        if alg.get("passed") is False:
            return False

        # Check numerical sanity
        num = details.get("numerical_sanity") or {}
        if num.get("passed") is False:
            return False

        # Check programmatic dimensional analysis (NEW - catches dimensional errors early)
        prog_dim = details.get("programmatic_dimensional_check") or {}
        if prog_dim.get("passed") is False:
            return False

        return True

    def _generate_feedback(self, details: Dict[str, Any]) -> str:
        """Generate feedback for failed validation."""
        feedback_parts = ["SYMBOLIC MATH CHECK FAILED:\n"]

        # Parse error - expression couldn't be parsed
        if not details.get("expression_parsed", False):
            parse_error = details.get("parse_error", "Unknown parse error")
            feedback_parts.append(f"Could not parse the answer expression: {parse_error}\n")
            feedback_parts.append("Please ensure the answer uses valid mathematical notation that can be symbolically verified.\n")
            feedback_parts.append("Common issues: malformed LaTeX, unbalanced brackets, undefined functions.\n")

        # Algebraic issues
        alg = details.get("algebraic_check") or {}
        if alg.get("issues"):
            feedback_parts.append("Algebraic issues found:\n")
            for issue in alg["issues"]:
                feedback_parts.append(f"  - {issue.get('type', 'Unknown')}: {issue.get('note', '')}\n")

        # Numerical issues
        num = details.get("numerical_sanity") or {}
        if num.get("issues"):
            feedback_parts.append("Numerical issues found:\n")
            for issue in num["issues"]:
                feedback_parts.append(f"  - {issue}\n")

        # Programmatic dimensional analysis error (NEW)
        prog_dim = details.get("programmatic_dimensional_check") or {}
        if prog_dim.get("passed") is False:
            feedback_parts.append("DIMENSIONAL ANALYSIS ERROR:\n")
            feedback_parts.append(f"  - Actual dimensions: {prog_dim.get('actual_dimensions', 'unknown')}\n")
            feedback_parts.append(f"  - Expected dimensions: {prog_dim.get('expected_dimensions', 'unknown')}\n")
            if prog_dim.get("error"):
                feedback_parts.append(f"  - {prog_dim['error']}\n")
            feedback_parts.append("  - YOU MUST fix this - the answer has WRONG UNITS.\n")
            feedback_parts.append("  - Re-derive from first principles and track dimensions at every step.\n")

        # Limiting case warnings
        limits = details.get("limiting_cases", [])
        for lim in limits:
            if lim.get("is_infinite"):
                feedback_parts.append(
                    f"Warning: Expression diverges as {lim['variable']} -> {lim['limit']}\n"
                )

        return "".join(feedback_parts)


def validate_expression_equivalence(expr1: str, expr2: str) -> Tuple[bool, str]:
    """
    Check if two expressions are mathematically equivalent.

    Useful for comparing student answer to reference answer.

    Returns:
        Tuple of (equivalent, explanation)
    """
    if not SYMPY_AVAILABLE:
        return True, "SymPy not available - cannot verify equivalence"

    try:
        validator = SymbolicMathValidator()

        parsed1 = validator._parse_physics_expression(expr1)
        parsed2 = validator._parse_physics_expression(expr2)

        if parsed1 is None or parsed2 is None:
            return True, "Could not parse one or both expressions"

        # Try to simplify the difference
        diff = simplify(parsed1 - parsed2)

        if diff == 0:
            return True, "Expressions are identical after simplification"

        # Try expanding and simplifying
        diff_expanded = simplify(sp.expand(parsed1) - sp.expand(parsed2))
        if diff_expanded == 0:
            return True, "Expressions are equivalent after expansion"

        # Check if ratio is a constant
        ratio = simplify(parsed1 / parsed2)
        if ratio.is_number and ratio != 0:
            return False, f"Expressions differ by a constant factor: {ratio}"

        return False, f"Expressions appear different: {expr1} vs {expr2}"

    except Exception as e:
        return True, f"Could not verify equivalence: {e}"
