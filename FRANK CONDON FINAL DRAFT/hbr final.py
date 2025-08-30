import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

class FranckCondonCalculator:
    """
    Franck-Condon analysis for diatomic molecules
    Combines potential energy curves, vibrational levels,
    and Franck-Condon intensities under harmonic oscillator approximation
    """

    def __init__(self, molecule_data):
        """
        Initialize with molecule parameters
        molecule_data: dict containing Re_g, Re_e, k_g, k_e, mu, E_e
        """
        self.Re_g = molecule_data['Re_g']  # Ground state equilibrium distance (Å)
        self.Re_e = molecule_data['Re_e']  # Excited state equilibrium distance (Å)
        self.k_g = molecule_data['k_g']    # Ground state force constant (eV/Å²)
        self.k_e = molecule_data['k_e']    # Excited state force constant (eV/Å²)
        self.mu = molecule_data['mu']      # Reduced mass (kg)
        self.E_e = molecule_data['E_e']    # Vertical energy shift (eV)

        # Convert units and calculate frequencies
        self._calculate_frequencies()
        self._calculate_alpha_parameters()

    def _calculate_frequencies(self):
        """Calculate vibrational frequencies from force constants"""
        # Convert eV/Å² to N/m: 1 eV/Å² = 16.02176634 N/m
        k_g_SI = self.k_g * 16.02176634
        k_e_SI = self.k_e * 16.02176634

        # Angular frequencies ω = √(k/μ)
        self.omega_g = np.sqrt(k_g_SI / self.mu)
        self.omega_e = np.sqrt(k_e_SI / self.mu)

        print(f"Ground state frequency: ωg = {self.omega_g:.2e} rad/s")
        print(f"Excited state frequency: ωe = {self.omega_e:.2e} rad/s")

    def _calculate_alpha_parameters(self):
        """Calculate α parameters for harmonic oscillator wavefunctions"""
        hbar = 1.054571817e-34  # J·s

        self.alpha_g = np.sqrt(self.mu * self.omega_g / hbar) * 1e-10  # Å⁻¹
        self.alpha_e = np.sqrt(self.mu * self.omega_e / hbar) * 1e-10  # Å⁻¹

        print(f"αg = {self.alpha_g:.2e} Å⁻¹")
        print(f"αe = {self.alpha_e:.2e} Å⁻¹")

    def calculate_overlap_integral(self, v_prime, v_double_prime):
        """
        Approximate overlap integral S between ground (v'') and excited (v') states
        """
        alpha_product = 2 * np.sqrt(self.alpha_g * self.alpha_e)
        alpha_sum = self.alpha_g + self.alpha_e
        delta_R_sq = (self.Re_e - self.Re_g)**2

        exp_factor = np.exp(-self.alpha_g * self.alpha_e * delta_R_sq / (2 * alpha_sum))

        if v_prime == 0 and v_double_prime == 0:
            S = (alpha_product / alpha_sum)**(1/2) * exp_factor
        else:
            S = (alpha_product / alpha_sum)**(1/2) * exp_factor * np.exp(-(v_prime + v_double_prime) * 0.1)

        return S

    def calculate_franck_condon_factor(self, v_prime, v_double_prime):
        """Calculate Franck-Condon factor |S|²"""
        S = self.calculate_overlap_integral(v_prime, v_double_prime)
        return abs(S)**2

    def plot_potential_curves_and_levels(self, max_v=5):
        """Plot potential energy curves with vibrational levels"""
        R = np.linspace(-1, 4, 1000)

        # Potential energy curves
        V_g = 0.5 * self.k_g * (R - self.Re_g)**2
        V_e = self.E_e + 0.5 * self.k_e * (R - self.Re_e)**2

        # Vibrational energy levels
        hbar = 1.054571817e-34  # J·s
        eV_to_J = 1.602176634e-19
        E_vib_g = [(v + 0.5) * hbar * self.omega_g / eV_to_J for v in range(max_v)]
        E_vib_e = [(v + 0.5) * hbar * self.omega_e / eV_to_J + self.E_e for v in range(max_v)]

        plt.figure(figsize=(12, 8))
        plt.plot(R, V_g, 'b-', linewidth=2, label='Ground State')
        plt.plot(R, V_e, 'r-', linewidth=2, label='Excited State')

        # Mark equilibrium distances
        plt.axvline(x=self.Re_g, color='blue', linestyle='--', alpha=0.7, label=f'Re(g) = {self.Re_g} Å')
        plt.axvline(x=self.Re_e, color='red', linestyle='--', alpha=0.7, label=f'Re(e) = {self.Re_e} Å')

        # Vibrational levels
        for v in range(max_v):
            plt.axhline(y=E_vib_g[v], color='blue', linestyle='--', alpha=0.6)
            plt.axhline(y=E_vib_e[v], color='red', linestyle='--', alpha=0.6)
            plt.text(-1, E_vib_g[v], f"v''={v}", fontsize=8, color='blue')
            plt.text(4, E_vib_e[v], f"v'={v}", fontsize=8, color='red')

        plt.xlabel('Internuclear Distance R (Å)')
        plt.ylabel('Potential Energy (eV)')
        plt.title('Potential Energy Curves with Vibrational Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-1, 4)
        plt.ylim(0, max(V_e) * 1.1)
        plt.savefig("HBr PE.png")
        plt.show()

    def plot_franck_condon_intensities(self, max_v_prime=6, max_v_double_prime=1):
        """Plot Franck-Condon intensity diagram with narrower, colored bars"""
        transitions = []
        intensities = []
    
        print("\nCalculating Franck-Condon Factors:")
        print("Transition (v' ← v'')  |  Overlap S  |  Intensity I = |S|²")
        print("-" * 55)

        for v_pp in range(max_v_double_prime):
            for v_p in range(max_v_prime):
                S = self.calculate_overlap_integral(v_p, v_pp)
                I = abs(S)**2
                transitions.append(f"{v_p} ← {v_pp}")
                intensities.append(I)
                print(f"    {v_p} ← {v_pp}        |   {S:.4f}    |    {I:.4f}")

        plt.figure(figsize=(12, 6))
        bars = plt.bar(transitions, intensities, width=0.3, color='orange', edgecolor='darkred', alpha=0.8)

        # Label intensities on top of bars
        for bar, intensity in zip(bars, intensities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{intensity:.3f}', ha='center', va='bottom', fontsize=10)

        plt.xlabel("Vibrational Transition (v' ← v'' )")
        plt.ylabel('Franck-Condon Intensity |S|²')
        plt.title('Franck-Condon Intensity Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("HBr ITENSITY.png")
        plt.show()

        return transitions, intensities

def main():
    hbr_data = {
        'Re_g': 1.414,
        'Re_e': 1.65,
        'k_g': 3.7,
        'k_e': 3,
        'mu': 1.65e-27,
        'E_e': 7.5
    }

    print("Franck-Condon Analysis for HBr")
    print("=" * 40)
    print(f"Ground state Re: {hbr_data['Re_g']} Å")
    print(f"Excited state Re: {hbr_data['Re_e']} Å")
    print(f"Ground state k: {hbr_data['k_g']} eV/Å²")
    print(f"Excited state k: {hbr_data['k_e']} eV/Å²")
    print(f"Reduced mass: {hbr_data['mu']:.2e} kg")
    print(f"Vertical energy shift Ee: {hbr_data['E_e']} eV\n")

    fc_calc = FranckCondonCalculator(hbr_data)

    print("Plotting potential energy curves and vibrational levels...")
    fc_calc.plot_potential_curves_and_levels()

    print("Calculating Franck-Condon intensities...")
    transitions, intensities = fc_calc.plot_franck_condon_intensities()

    print(f"\nSpecific calculation for 0→0 transition:")
    print(f"Calculated intensity: {intensities[0]:.3f}")
    print(f"Expected (example): 0.283")
    print(f"Relative agreement: {intensities[0]/0.283:.1%}")


if __name__ == "__main__":
    main()



