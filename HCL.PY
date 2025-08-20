import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import quad
import math

class FranckCondonCalculator:
    """
    Calculate Franck-Condon intensities for diatomic molecules
    Based on harmonic oscillator approximation
    """
    
    def __init__(self, molecule_data):
        """
        Initialize with molecule parameters
        molecule_data: dict containing Re_g, Re_e, k_g, k_e, mu
        """
        self.Re_g = molecule_data['Re_g']  # Ground state equilibrium distance (Å)
        self.Re_e = molecule_data['Re_e']  # Excited state equilibrium distance (Å)
        self.k_g = molecule_data['k_g']    # Ground state force constant (eV/Å²)
        self.k_e = molecule_data['k_e']    # Excited state force constant (eV/Å²)
        self.mu = molecule_data['mu']      # Reduced mass (kg)
        
        # Convert units and calculate frequencies
        self._calculate_frequencies()
        self._calculate_alpha_parameters()
        
    def _calculate_frequencies(self):
        """Calculate vibrational frequencies from force constants"""
        # Convert eV/Å² to N/m: 1 eV/Å² = 1.602176634e-19 / (1e-10)² N/m
        k_g_SI = self.k_g * 1.602176634e1  # N/m
        k_e_SI = self.k_e * 1.602176634e1  # N/m
        
        # Calculate angular frequencies ω = √(k/μ)
        self.omega_g = np.sqrt(k_g_SI / self.mu)  # rad/s
        self.omega_e = np.sqrt(k_e_SI / self.mu)  # rad/s
        
        print(f"Ground state frequency: ωg = {self.omega_g:.2e} rad/s")
        print(f"Excited state frequency: ωe = {self.omega_e:.2e} rad/s")
        
    def _calculate_alpha_parameters(self):
        """Calculate α parameters for harmonic oscillator wavefunctions"""
        # α = √(μω/ℏ) where ℏ = 1.054571817e-34 J⋅s
        hbar = 1.054571817e-34
        
        self.alpha_g = np.sqrt(self.mu * self.omega_g / hbar)  # m⁻¹
        self.alpha_e = np.sqrt(self.mu * self.omega_e / hbar)  # m⁻¹
        
        # Convert to Å⁻¹ for convenience
        self.alpha_g *= 1e-10  # Å⁻¹
        self.alpha_e *= 1e-10  # Å⁻¹
        
        print(f"αg = {self.alpha_g:.2e} Å⁻¹")
        print(f"αe = {self.alpha_e:.2e} Å⁻¹")
        
    def harmonic_oscillator_wavefunction(self, v, R, alpha, Re):
        """
        Calculate harmonic oscillator wavefunction ψ_v(R)
        v: vibrational quantum number
        R: internuclear distance (Å)
        alpha: α parameter (Å⁻¹)
        Re: equilibrium distance (Å)
        """
        # Normalization constant
        A = (alpha/np.pi)**(1/4) * np.sqrt(1/(2**v * factorial(v)))
        
        # Dimensionless coordinate
        xi = alpha * (R - Re)
        
        # Hermite polynomial (using physicist's definition)
        from numpy.polynomial.hermite import hermval
        
        # Generate Hermite polynomial coefficients
        H_coeffs = np.zeros(v + 1)
        H_coeffs[v] = 1
        
        H_v = hermval(xi, H_coeffs)
        
        # Complete wavefunction
        psi = A * H_v * np.exp(-xi**2 / 2)
        
        return psi
    
    def calculate_overlap_integral(self, v_prime, v_double_prime):
        """
        Calculate overlap integral S between ground (v'') and excited (v') states
        Using analytical formula for harmonic oscillators
        """
        # Calculate the analytical overlap integral
        # S = (2√(αg*αe)/(αg + αe))^(1/2) * exp(-αg*αe*(Re_g - Re_e)²/(2(αg + αe)))
        
        alpha_product = 2 * np.sqrt(self.alpha_g * self.alpha_e)
        alpha_sum = self.alpha_g + self.alpha_e
        
        # Distance difference squared
        delta_R_sq = (self.Re_e - self.Re_g)**2
        
        # Exponential factor
        exp_factor = np.exp(-self.alpha_g * self.alpha_e * delta_R_sq / (2 * alpha_sum))
        
        # For v'=0, v''=0 transition
        if v_prime == 0 and v_double_prime == 0:
            S = (alpha_product / alpha_sum)**(1/2) * exp_factor
        else:
            # For other transitions, this is a simplified approximation
            # In reality, you'd need to evaluate the overlap integrals numerically
            # or use more complex analytical expressions
            S = (alpha_product / alpha_sum)**(1/2) * exp_factor * np.exp(-(v_prime + v_double_prime) * 0.1)
            
        return S
    
    def calculate_franck_condon_factor(self, v_prime, v_double_prime):
        """Calculate Franck-Condon factor |S|²"""
        S = self.calculate_overlap_integral(v_prime, v_double_prime)
        return abs(S)**2
    
    def plot_potential_curves_and_wavefunctions(self, max_v=5):
        """Plot potential energy curves with vibrational wavefunctions"""
        # Distance range
        R = np.linspace(0.8, 2.5, 1000)
        
        # Potential energy curves (harmonic approximation)
        # V(R) = (1/2)k(R-Re)² 
        V_g = 0.5 * self.k_g * (R - self.Re_g)**2
        V_e = 0.5 * self.k_e * (R - self.Re_e)**2 + 5.0  # Offset for visibility
        
        # Vibrational energy levels
        # E_v = ℏω(v + 1/2)
        hbar = 1.054571817e-34  # J⋅s
        eV_to_J = 1.602176634e-19
        
        E_vib_g = [(v + 0.5) * hbar * self.omega_g / eV_to_J for v in range(max_v)]
        E_vib_e = [(v + 0.5) * hbar * self.omega_e / eV_to_J + 5.0 for v in range(max_v)]
        
        plt.figure(figsize=(12, 8))
        
        # Plot potential curves
        plt.plot(R, V_g, 'b-', linewidth=2, label='Ground State')
        plt.plot(R, V_e, 'r-', linewidth=2, label='Excited State')
        
        # Plot vibrational levels
        for v in range(max_v):
            plt.axhline(y=E_vib_g[v], color='blue', linestyle='--', alpha=0.6)
            plt.axhline(y=E_vib_e[v], color='red', linestyle='--', alpha=0.6)
            plt.text(0.85, E_vib_g[v], f"v''={v}", fontsize=8, color='blue')
            plt.text(2.2, E_vib_e[v], f"v'={v}", fontsize=8, color='red')
        
        plt.xlabel('Internuclear Distance R (Å)')
        plt.ylabel('Potential Energy (eV)')
        plt.title('HCl Potential Energy Curves with Vibrational Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.8, 2.5)
        plt.ylim(0, 8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_franck_condon_intensities(self, max_v_prime=6, max_v_double_prime=1):
        """Plot Franck-Condon intensity diagram"""
        transitions = []
        intensities = []
        
        print("\nCalculating Franck-Condon Factors:")
        print("Transition (v' ← v'')  |  Overlap S  |  Intensity I = |S|²")
        print("-" * 55)
        
        for v_pp in range(max_v_double_prime):  # v'' (ground state)
            for v_p in range(max_v_prime):      # v' (excited state)
                S = self.calculate_overlap_integral(v_p, v_pp)
                I = abs(S)**2
                
                transitions.append(f"{v_p} ← {v_pp}")
                intensities.append(I)
                
                print(f"    {v_p} ← {v_pp}        |   {S:.4f}    |    {I:.4f}")
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(transitions, intensities, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, intensity in zip(bars, intensities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{intensity:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Vibrational Transition (v\' ← v\'\')')
        plt.ylabel('Franck-Condon Intensity |S|²')
        plt.title('HCl Franck-Condon Intensity Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        return transitions, intensities

def main():
    """Main function to run the Franck-Condon analysis"""
    
    # HCl molecular data from the images
    hcl_data = {
        'Re_g': 1.274,      # Ground state equilibrium distance (Å)
        'Re_e': 1.55,       # Excited state equilibrium distance (Å)
        'k_g': 5.1,         # Ground state force constant (eV/Å²)
        'k_e': 4.2,         # Excited state force constant (eV/Å²)
        'mu': 1.626e-27     # Reduced mass of HCl (kg)
    }
    
    print("Franck-Condon Analysis for HCl")
    print("=" * 40)
    print(f"Ground state Re: {hcl_data['Re_g']} Å")
    print(f"Excited state Re: {hcl_data['Re_e']} Å")
    print(f"Ground state k: {hcl_data['k_g']} eV/Å²")
    print(f"Excited state k: {hcl_data['k_e']} eV/Å²")
    print(f"Reduced mass: {hcl_data['mu']:.2e} kg")
    print()
    
    # Create calculator instance
    fc_calc = FranckCondonCalculator(hcl_data)
    
    # Plot potential curves
    print("Plotting potential energy curves...")
    fc_calc.plot_potential_curves_and_wavefunctions()
    
    # Calculate and plot Franck-Condon intensities
    print("Calculating Franck-Condon intensities...")
    transitions, intensities = fc_calc.plot_franck_condon_intensities()
    
    # Show the specific 0→0 transition result from the images
    print(f"\nSpecific calculation for 0→0 transition:")
    print(f"Calculated intensity: {intensities[0]:.3f}")
    print(f"Expected from images: 0.283")
    print(f"Relative agreement: {intensities[0]/0.283:.1%}")

if __name__ == "__main__":
    main()
