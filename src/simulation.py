"""
EEZO Unboxing Experience Compound Growth Simulator

Simulates how investment in unboxing experience quality
compounds through repurchase rates and UGC generation
"""

from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os


@dataclass
class SimulationParams:
    """Simulation Parameters"""
    # Unboxing experience quality (1-10)
    experience_quality: float = 5.0
    
    # Repurchase
    base_repurchase_rate: float = 0.15  # Base repurchase rate
    quality_to_repurchase: float = 0.03  # Repurchase rate increase per quality point
    
    # UGC
    base_ugc_rate: float = 0.02  # Base UGC posting rate
    quality_to_ugc: float = 0.015  # UGC rate increase per quality point
    
    # New customer acquisition
    monthly_visitors: int = 1000  # Monthly site visitors
    base_cvr: float = 0.02  # Base CVR
    ugc_to_cvr_boost: float = 0.0005  # CVR improvement per UGC post
    max_cvr_boost: float = 0.03  # Max CVR improvement cap
    
    # Initial conditions
    initial_customers: int = 500  # Initial customer base
    initial_ugc_count: int = 10  # Initial UGC count
    
    # Simulation period
    months: int = 24


@dataclass
class MonthlyState:
    """Monthly State"""
    month: int
    total_customers: int
    active_customers: int  # Customers who purchased in last 12 months
    new_customers: int
    repeat_purchases: int
    ugc_count: int
    new_ugc: int
    cvr: float
    repurchase_rate: float


def calculate_rates(params: SimulationParams, ugc_count: int) -> Tuple[float, float, float]:
    """Calculate rates based on quality and accumulated UGC"""
    
    # Repurchase rate = base + (quality - 5) * coefficient
    quality_bonus = (params.experience_quality - 5) * params.quality_to_repurchase
    repurchase_rate = min(0.6, max(0.05, params.base_repurchase_rate + quality_bonus))
    
    # UGC posting rate = base + (quality - 5) * coefficient
    ugc_bonus = (params.experience_quality - 5) * params.quality_to_ugc
    ugc_rate = min(0.15, max(0.01, params.base_ugc_rate + ugc_bonus))
    
    # CVR = base + UGC accumulation effect (capped)
    ugc_boost = min(params.max_cvr_boost, ugc_count * params.ugc_to_cvr_boost)
    cvr = params.base_cvr + ugc_boost
    
    return repurchase_rate, ugc_rate, cvr


def simulate(params: SimulationParams) -> List[MonthlyState]:
    """Run monthly simulation"""
    
    history: List[MonthlyState] = []
    
    total_customers = params.initial_customers
    active_customers = params.initial_customers
    ugc_count = params.initial_ugc_count
    
    recent_new_customers = [0] * 12
    recent_new_customers[0] = params.initial_customers
    
    for month in range(1, params.months + 1):
        repurchase_rate, ugc_rate, cvr = calculate_rates(params, ugc_count)
        
        new_customers = int(params.monthly_visitors * cvr)
        total_customers += new_customers
        
        repeat_purchases = int(active_customers * repurchase_rate)
        monthly_purchasers = new_customers + repeat_purchases
        
        new_ugc = int(monthly_purchasers * ugc_rate)
        ugc_count += new_ugc
        
        recent_new_customers.pop(0)
        recent_new_customers.append(new_customers)
        active_customers = sum(recent_new_customers) + int(total_customers * 0.3)
        active_customers = min(active_customers, total_customers)
        
        state = MonthlyState(
            month=month,
            total_customers=total_customers,
            active_customers=active_customers,
            new_customers=new_customers,
            repeat_purchases=repeat_purchases,
            ugc_count=ugc_count,
            new_ugc=new_ugc,
            cvr=cvr,
            repurchase_rate=repurchase_rate
        )
        history.append(state)
    
    return history


def compare_scenarios(months: int = 24) -> Tuple[List[MonthlyState], List[MonthlyState]]:
    """Compare no-investment vs investment scenarios"""
    
    params_baseline = SimulationParams(experience_quality=5.0, months=months)
    params_invested = SimulationParams(experience_quality=8.0, months=months)
    
    baseline = simulate(params_baseline)
    invested = simulate(params_invested)
    
    return baseline, invested


def plot_comparison(baseline: List[MonthlyState], invested: List[MonthlyState], 
                    output_path: str = "outputs/compound_growth.png"):
    """Generate comparison chart"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    months = [s.month for s in baseline]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Unboxing Experience Investment: Compound Growth Effect\n(Quality 5 vs Quality 8)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Total Customers
    ax1 = axes[0, 0]
    ax1.plot(months, [s.total_customers for s in baseline], 'b-', label='Baseline (Q=5)', linewidth=2)
    ax1.plot(months, [s.total_customers for s in invested], 'r-', label='Invested (Q=8)', linewidth=2)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Total Customers')
    ax1.set_title('Total Customer Growth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    final_diff = invested[-1].total_customers - baseline[-1].total_customers
    final_ratio = invested[-1].total_customers / baseline[-1].total_customers
    ax1.annotate(f'+{final_diff:,}\n({final_ratio:.1%})', 
                 xy=(months[-1], invested[-1].total_customers),
                 xytext=(months[-1]-3, invested[-1].total_customers*0.9),
                 fontsize=10, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    # 2. UGC Accumulation
    ax2 = axes[0, 1]
    ax2.plot(months, [s.ugc_count for s in baseline], 'b-', label='Baseline', linewidth=2)
    ax2.plot(months, [s.ugc_count for s in invested], 'r-', label='Invested', linewidth=2)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Total UGC Count')
    ax2.set_title('UGC Accumulation (Trust Signal)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. CVR Progression
    ax3 = axes[1, 0]
    ax3.plot(months, [s.cvr * 100 for s in baseline], 'b-', label='Baseline', linewidth=2)
    ax3.plot(months, [s.cvr * 100 for s in invested], 'r-', label='Invested', linewidth=2)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('CVR (%)')
    ax3.set_title('CVR Progression (UGC Effect)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Monthly Purchases
    ax4 = axes[1, 1]
    new_baseline = [s.new_customers for s in baseline]
    repeat_baseline = [s.repeat_purchases for s in baseline]
    new_invested = [s.new_customers for s in invested]
    repeat_invested = [s.repeat_purchases for s in invested]
    
    width = 0.35
    selected_months = [6, 12, 18, 24]
    selected_idx = [m-1 for m in selected_months]
    
    x_selected = np.arange(len(selected_months))
    ax4.bar(x_selected - width/2, [new_baseline[i] + repeat_baseline[i] for i in selected_idx], 
            width, label='Baseline', color='blue', alpha=0.7)
    ax4.bar(x_selected + width/2, [new_invested[i] + repeat_invested[i] for i in selected_idx], 
            width, label='Invested', color='red', alpha=0.7)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Monthly Purchases (New + Repeat)')
    ax4.set_title('Monthly Purchase Comparison')
    ax4.set_xticks(x_selected)
    ax4.set_xticklabels([f'M{m}' for m in selected_months])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved: {output_path}")


def print_summary(baseline: List[MonthlyState], invested: List[MonthlyState]):
    """Print summary"""
    
    print("\n" + "="*60)
    print("COMPOUND GROWTH SIMULATION RESULTS")
    print("="*60)
    
    print("\n## Scenario Comparison (24 months)")
    print("-"*50)
    print(f"{'Metric':<25} {'Baseline':>12} {'Invested':>12} {'Diff':>10}")
    print("-"*50)
    
    b, i = baseline[-1], invested[-1]
    
    metrics = [
        ("Total Customers", b.total_customers, i.total_customers),
        ("Total UGC Count", b.ugc_count, i.ugc_count),
        ("Final CVR", f"{b.cvr:.2%}", f"{i.cvr:.2%}"),
        ("Final Repurchase Rate", f"{b.repurchase_rate:.2%}", f"{i.repurchase_rate:.2%}"),
    ]
    
    for name, val_b, val_i in metrics:
        if isinstance(val_b, int):
            diff = val_i - val_b
            print(f"{name:<25} {val_b:>12,} {val_i:>12,} {diff:>+10,}")
        else:
            print(f"{name:<25} {val_b:>12} {val_i:>12}")
    
    print("-"*50)
    
    customer_ratio = i.total_customers / b.total_customers
    ugc_ratio = i.ugc_count / b.ugc_count
    
    print(f"\n## Compound Effect Summary")
    print(f"  Customer Growth: {customer_ratio:.1%} (+{customer_ratio-1:.1%})")
    print(f"  UGC Growth:      {ugc_ratio:.1%} (+{ugc_ratio-1:.1%})")
    
    print("\n## Compound Loop Structure")
    print("  1. High Quality -> Repurchase Rate +9pt (15% -> 24%)")
    print("  2. High Quality -> UGC Rate +4.5pt (2% -> 6.5%)")
    print("  3. UGC Accumulation -> CVR Up -> More New Customers")
    print("  4. More Customers x High Repurchase = More UGC")
    print("  -> This loop creates the 'compound' effect")
    
    print("\n" + "="*60)


def main():
    """Main execution"""
    print("EEZO Unboxing Experience Compound Growth Simulator")
    print("-"*50)
    
    baseline, invested = compare_scenarios(months=24)
    plot_comparison(baseline, invested)
    print_summary(baseline, invested)


if __name__ == "__main__":
    main()
