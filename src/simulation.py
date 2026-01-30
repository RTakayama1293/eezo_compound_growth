"""
EEZO 信頼醸成による複利成長シミュレーター（厳密版）

5年間の売上への影響をシミュレート：
- コホート分析による正確な再購入追跡
- 信頼シグナル（UGC）の蓄積効果
- LTV・売上の複利的成長
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'sans-serif']


@dataclass
class SimulationParams:
    """シミュレーションパラメータ"""

    # 信頼・体験品質スコア (1-10)
    trust_quality: float = 5.0

    # 顧客獲得
    monthly_visitors: int = 2300          # 月間サイト訪問者数
    base_cvr: float = 0.015               # 基準CVR (1.5%)

    # 信頼効果 → CVR改善（対数関数で逓減）
    ugc_cvr_coefficient: float = 0.003    # UGC 1件あたりのCVR寄与（逓減前）
    max_cvr: float = 0.08                 # CVR上限 (8%)

    # 再購入（コホート別月次確率）
    base_monthly_repurchase_prob: float = 0.08   # 基準月次再購入確率
    quality_repurchase_bonus: float = 0.015       # 品質1ptあたりの再購入確率増加

    # 顧客離脱（チャーン）
    monthly_churn_rate: float = 0.03      # 月次離脱率
    quality_churn_reduction: float = 0.005 # 品質1ptあたりの離脱率減少

    # UGC生成
    base_ugc_rate: float = 0.01           # 基準UGC投稿率（購入あたり）
    quality_ugc_bonus: float = 0.008      # 品質1ptあたりのUGC率増加
    ugc_decay_rate: float = 0.02          # UGC月次減衰率（古いUGCの効果減少）

    # 紹介効果
    base_referral_rate: float = 0.02      # 基準紹介率
    quality_referral_bonus: float = 0.01  # 品質1ptあたりの紹介率増加

    # 売上パラメータ
    average_order_value: float = 8000     # 平均客単価（円）
    repeat_order_premium: float = 1.15    # リピーターの客単価係数

    # 初期条件
    initial_customers: int = 100
    initial_ugc: int = 10

    # シミュレーション期間
    months: int = 60  # 5年


@dataclass
class Cohort:
    """顧客コホート（獲得月別の顧客グループ）"""
    acquired_month: int
    initial_size: int
    current_size: float
    total_purchases: int = 0
    total_revenue: float = 0


@dataclass
class MonthlyMetrics:
    """月次メトリクス"""
    month: int

    # 顧客数
    total_active_customers: int
    new_customers: int
    churned_customers: int
    referred_customers: int

    # 購入
    new_purchases: int
    repeat_purchases: int
    total_purchases: int

    # 売上
    new_revenue: float
    repeat_revenue: float
    total_revenue: float
    cumulative_revenue: float

    # 信頼指標
    effective_ugc: float        # 減衰後の有効UGC数
    total_ugc_generated: int
    new_ugc: int
    cvr: float

    # 率
    monthly_repurchase_rate: float
    churn_rate: float

    # LTV関連
    avg_customer_ltv: float


class TrustCompoundSimulator:
    """信頼醸成複利シミュレーター"""

    def __init__(self, params: SimulationParams):
        self.params = params
        self.cohorts: List[Cohort] = []
        self.history: List[MonthlyMetrics] = []
        self.ugc_history: List[Tuple[int, int]] = []  # (month, count)

        # 品質による調整値を事前計算
        quality_delta = params.trust_quality - 5.0
        self.repurchase_prob = min(0.25, max(0.02,
            params.base_monthly_repurchase_prob + quality_delta * params.quality_repurchase_bonus))
        self.churn_rate = max(0.005,
            params.monthly_churn_rate - quality_delta * params.quality_churn_reduction)
        self.ugc_rate = min(0.15, max(0.005,
            params.base_ugc_rate + quality_delta * params.quality_ugc_bonus))
        self.referral_rate = min(0.10, max(0.01,
            params.base_referral_rate + quality_delta * params.quality_referral_bonus))

    def calculate_effective_ugc(self, current_month: int) -> float:
        """減衰を考慮した有効UGC数を計算"""
        effective = 0.0
        for month, count in self.ugc_history:
            age_months = current_month - month
            decay = np.exp(-self.params.ugc_decay_rate * age_months)
            effective += count * decay
        return effective

    def calculate_cvr(self, effective_ugc: float) -> float:
        """UGC蓄積量からCVRを計算（対数関数で逓減効果）"""
        # CVR = base + coefficient * log(1 + ugc)
        ugc_effect = self.params.ugc_cvr_coefficient * np.log1p(effective_ugc)
        cvr = self.params.base_cvr + ugc_effect
        return min(self.params.max_cvr, cvr)

    def simulate_month(self, month: int) -> MonthlyMetrics:
        """1ヶ月分のシミュレーション"""

        p = self.params

        # 有効UGCとCVR計算
        effective_ugc = self.calculate_effective_ugc(month)
        cvr = self.calculate_cvr(effective_ugc)

        # 新規顧客獲得
        new_from_traffic = int(p.monthly_visitors * cvr)

        # 既存顧客からの紹介
        total_active = sum(c.current_size for c in self.cohorts)
        referred = int(total_active * self.referral_rate / 12)  # 年率を月率に

        new_customers = new_from_traffic + referred

        # 新規コホート追加
        if new_customers > 0:
            new_cohort = Cohort(
                acquired_month=month,
                initial_size=new_customers,
                current_size=float(new_customers),
                total_purchases=new_customers,
                total_revenue=new_customers * p.average_order_value
            )
            self.cohorts.append(new_cohort)

        new_revenue = new_customers * p.average_order_value

        # 既存コホートの再購入とチャーン
        repeat_purchases = 0
        repeat_revenue = 0.0
        churned_total = 0

        for cohort in self.cohorts:
            if cohort.acquired_month == month:
                continue  # 今月獲得したコホートはスキップ

            # チャーン計算
            churned = cohort.current_size * self.churn_rate
            cohort.current_size -= churned
            churned_total += churned

            if cohort.current_size < 1:
                continue

            # 再購入計算（月次確率）
            purchases = int(cohort.current_size * self.repurchase_prob)
            if purchases > 0:
                revenue = purchases * p.average_order_value * p.repeat_order_premium
                repeat_purchases += purchases
                repeat_revenue += revenue
                cohort.total_purchases += purchases
                cohort.total_revenue += revenue

        total_purchases = new_customers + repeat_purchases
        total_revenue = new_revenue + repeat_revenue

        # UGC生成
        new_ugc = int(total_purchases * self.ugc_rate)
        if new_ugc > 0:
            self.ugc_history.append((month, new_ugc))

        total_ugc = sum(count for _, count in self.ugc_history)

        # 累計売上
        prev_cumulative = self.history[-1].cumulative_revenue if self.history else 0
        cumulative_revenue = prev_cumulative + total_revenue

        # アクティブ顧客数
        total_active_customers = int(sum(c.current_size for c in self.cohorts))

        # 平均LTV計算
        if total_active_customers > 0:
            total_ltv_revenue = sum(c.total_revenue for c in self.cohorts)
            total_customers_ever = sum(c.initial_size for c in self.cohorts)
            avg_ltv = total_ltv_revenue / total_customers_ever if total_customers_ever > 0 else 0
        else:
            avg_ltv = 0

        metrics = MonthlyMetrics(
            month=month,
            total_active_customers=total_active_customers,
            new_customers=new_customers,
            churned_customers=int(churned_total),
            referred_customers=referred,
            new_purchases=new_customers,
            repeat_purchases=repeat_purchases,
            total_purchases=total_purchases,
            new_revenue=new_revenue,
            repeat_revenue=repeat_revenue,
            total_revenue=total_revenue,
            cumulative_revenue=cumulative_revenue,
            effective_ugc=effective_ugc,
            total_ugc_generated=total_ugc,
            new_ugc=new_ugc,
            cvr=cvr,
            monthly_repurchase_rate=self.repurchase_prob,
            churn_rate=self.churn_rate,
            avg_customer_ltv=avg_ltv
        )

        self.history.append(metrics)
        return metrics

    def run(self) -> List[MonthlyMetrics]:
        """シミュレーション実行"""

        # 初期顧客をコホートとして追加
        if self.params.initial_customers > 0:
            initial_cohort = Cohort(
                acquired_month=0,
                initial_size=self.params.initial_customers,
                current_size=float(self.params.initial_customers),
                total_purchases=self.params.initial_customers,
                total_revenue=self.params.initial_customers * self.params.average_order_value
            )
            self.cohorts.append(initial_cohort)

        # 初期UGC
        if self.params.initial_ugc > 0:
            self.ugc_history.append((0, self.params.initial_ugc))

        # 月次シミュレーション
        for month in range(1, self.params.months + 1):
            self.simulate_month(month)

        return self.history


def run_comparison(months: int = 60) -> Tuple[List[MonthlyMetrics], List[MonthlyMetrics]]:
    """ベースラインと投資シナリオの比較"""

    # ベースライン（信頼品質 = 5）
    baseline_params = SimulationParams(trust_quality=5.0, months=months)
    baseline_sim = TrustCompoundSimulator(baseline_params)
    baseline = baseline_sim.run()

    # 投資シナリオ（信頼品質 = 8）
    invested_params = SimulationParams(trust_quality=8.0, months=months)
    invested_sim = TrustCompoundSimulator(invested_params)
    invested = invested_sim.run()

    return baseline, invested


def plot_results(baseline: List[MonthlyMetrics], invested: List[MonthlyMetrics],
                 output_path: str = "outputs/compound_growth.png"):
    """結果をプロット"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    months = [m.month for m in baseline]
    years = [m / 12 for m in months]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Trust-Based Compound Growth Simulation (5 Years)\nQuality 5 (Baseline) vs Quality 8 (Invested)',
                 fontsize=16, fontweight='bold')

    # 1. 累計売上
    ax1 = axes[0, 0]
    baseline_revenue = [m.cumulative_revenue / 1_000_000 for m in baseline]
    invested_revenue = [m.cumulative_revenue / 1_000_000 for m in invested]
    ax1.plot(years, baseline_revenue, 'b-', label='Baseline (Q=5)', linewidth=2.5)
    ax1.plot(years, invested_revenue, 'r-', label='Invested (Q=8)', linewidth=2.5)
    ax1.fill_between(years, baseline_revenue, invested_revenue, alpha=0.3, color='green')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Cumulative Revenue (Million Yen)')
    ax1.set_title('Cumulative Revenue')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}M'))

    diff_revenue = invested[-1].cumulative_revenue - baseline[-1].cumulative_revenue
    ax1.annotate(f'+{diff_revenue/1_000_000:,.1f}M yen\n({invested[-1].cumulative_revenue/baseline[-1].cumulative_revenue:.1%})',
                 xy=(5, invested_revenue[-1]), fontsize=11, color='red', fontweight='bold',
                 xytext=(4, invested_revenue[-1]*0.85),
                 arrowprops=dict(arrowstyle='->', color='red'))

    # 2. 月次売上
    ax2 = axes[0, 1]
    ax2.plot(years, [m.total_revenue / 1_000_000 for m in baseline], 'b-', label='Baseline', linewidth=2)
    ax2.plot(years, [m.total_revenue / 1_000_000 for m in invested], 'r-', label='Invested', linewidth=2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Monthly Revenue (Million Yen)')
    ax2.set_title('Monthly Revenue Trend')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. アクティブ顧客数
    ax3 = axes[0, 2]
    ax3.plot(years, [m.total_active_customers for m in baseline], 'b-', label='Baseline', linewidth=2)
    ax3.plot(years, [m.total_active_customers for m in invested], 'r-', label='Invested', linewidth=2)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Active Customers')
    ax3.set_title('Active Customer Base')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. CVR推移（信頼効果）
    ax4 = axes[1, 0]
    ax4.plot(years, [m.cvr * 100 for m in baseline], 'b-', label='Baseline', linewidth=2)
    ax4.plot(years, [m.cvr * 100 for m in invested], 'r-', label='Invested', linewidth=2)
    ax4.axhline(y=8.0, color='gray', linestyle='--', alpha=0.5, label='CVR Cap (8%)')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('CVR (%)')
    ax4.set_title('CVR Progression (Trust/UGC Effect)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 有効UGC数（信頼シグナル）
    ax5 = axes[1, 1]
    ax5.plot(years, [m.effective_ugc for m in baseline], 'b-', label='Baseline', linewidth=2)
    ax5.plot(years, [m.effective_ugc for m in invested], 'r-', label='Invested', linewidth=2)
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Effective UGC Count')
    ax5.set_title('Trust Signal Accumulation (Effective UGC)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 売上構成（新規 vs リピート）
    ax6 = axes[1, 2]
    year_points = [1, 2, 3, 4, 5]
    month_indices = [y * 12 - 1 for y in year_points]

    x = np.arange(len(year_points))
    width = 0.35

    new_b = [baseline[i].new_revenue / 1_000_000 for i in month_indices]
    repeat_b = [baseline[i].repeat_revenue / 1_000_000 for i in month_indices]
    new_i = [invested[i].new_revenue / 1_000_000 for i in month_indices]
    repeat_i = [invested[i].repeat_revenue / 1_000_000 for i in month_indices]

    ax6.bar(x - width/2, new_b, width/2, label='Baseline New', color='lightblue')
    ax6.bar(x - width/2, repeat_b, width/2, bottom=new_b, label='Baseline Repeat', color='blue')
    ax6.bar(x + width/2, new_i, width/2, label='Invested New', color='lightsalmon')
    ax6.bar(x + width/2, repeat_i, width/2, bottom=new_i, label='Invested Repeat', color='red')

    ax6.set_xlabel('Year')
    ax6.set_ylabel('Monthly Revenue (Million Yen)')
    ax6.set_title('Revenue Composition (New vs Repeat)')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'Y{y}' for y in year_points])
    ax6.legend(loc='upper left', fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Chart saved: {output_path}")


def print_detailed_report(baseline: List[MonthlyMetrics], invested: List[MonthlyMetrics],
                          baseline_params: SimulationParams, invested_params: SimulationParams):
    """詳細レポート出力"""

    print("\n" + "="*80)
    print("TRUST-BASED COMPOUND GROWTH SIMULATION - DETAILED REPORT (5 YEARS)")
    print("="*80)

    # パラメータ比較
    print("\n## Simulation Parameters")
    print("-"*60)
    print(f"{'Parameter':<35} {'Baseline':>12} {'Invested':>12}")
    print("-"*60)
    print(f"{'Trust Quality Score':<35} {baseline_params.trust_quality:>12.1f} {invested_params.trust_quality:>12.1f}")

    # 計算された率
    b_sim = TrustCompoundSimulator(baseline_params)
    i_sim = TrustCompoundSimulator(invested_params)
    print(f"{'Monthly Repurchase Probability':<35} {b_sim.repurchase_prob*100:>11.1f}% {i_sim.repurchase_prob*100:>11.1f}%")
    print(f"{'Monthly Churn Rate':<35} {b_sim.churn_rate*100:>11.1f}% {i_sim.churn_rate*100:>11.1f}%")
    print(f"{'UGC Generation Rate':<35} {b_sim.ugc_rate*100:>11.1f}% {i_sim.ugc_rate*100:>11.1f}%")
    print(f"{'Referral Rate (Annual)':<35} {b_sim.referral_rate*100:>11.1f}% {i_sim.referral_rate*100:>11.1f}%")

    # 年次サマリー
    print("\n## Annual Summary")
    print("-"*80)
    print(f"{'Year':<6} {'Baseline Revenue':>18} {'Invested Revenue':>18} {'Difference':>15} {'Growth':>10}")
    print("-"*80)

    for year in [1, 2, 3, 4, 5]:
        month_idx = year * 12 - 1
        b_rev = baseline[month_idx].cumulative_revenue
        i_rev = invested[month_idx].cumulative_revenue
        diff = i_rev - b_rev
        growth = (i_rev / b_rev - 1) * 100 if b_rev > 0 else 0
        print(f"Y{year:<5} {b_rev:>17,.0f} {i_rev:>17,.0f} {diff:>+14,.0f} {growth:>+9.1f}%")

    # 最終比較
    b_final = baseline[-1]
    i_final = invested[-1]

    print("\n## Final State Comparison (Month 60)")
    print("-"*60)
    print(f"{'Metric':<35} {'Baseline':>12} {'Invested':>12}")
    print("-"*60)

    metrics = [
        ("Cumulative Revenue (Yen)", f"{b_final.cumulative_revenue:,.0f}", f"{i_final.cumulative_revenue:,.0f}"),
        ("Monthly Revenue (Yen)", f"{b_final.total_revenue:,.0f}", f"{i_final.total_revenue:,.0f}"),
        ("Active Customers", f"{b_final.total_active_customers:,}", f"{i_final.total_active_customers:,}"),
        ("Total UGC Generated", f"{b_final.total_ugc_generated:,}", f"{i_final.total_ugc_generated:,}"),
        ("Effective UGC", f"{b_final.effective_ugc:,.1f}", f"{i_final.effective_ugc:,.1f}"),
        ("Final CVR", f"{b_final.cvr*100:.2f}%", f"{i_final.cvr*100:.2f}%"),
        ("Avg Customer LTV (Yen)", f"{b_final.avg_customer_ltv:,.0f}", f"{i_final.avg_customer_ltv:,.0f}"),
    ]

    for name, b_val, i_val in metrics:
        print(f"{name:<35} {b_val:>12} {i_val:>12}")

    # 複利効果の分析
    print("\n## Compound Effect Analysis")
    print("-"*60)

    revenue_ratio = i_final.cumulative_revenue / b_final.cumulative_revenue
    customer_ratio = i_final.total_active_customers / b_final.total_active_customers
    ugc_ratio = i_final.effective_ugc / b_final.effective_ugc if b_final.effective_ugc > 0 else 0

    print(f"5-Year Revenue Multiple:        {revenue_ratio:.2f}x (+{(revenue_ratio-1)*100:.1f}%)")
    print(f"Customer Base Multiple:         {customer_ratio:.2f}x (+{(customer_ratio-1)*100:.1f}%)")
    print(f"Trust Signal (UGC) Multiple:    {ugc_ratio:.2f}x (+{(ugc_ratio-1)*100:.1f}%)")

    # 投資対効果
    additional_revenue = i_final.cumulative_revenue - b_final.cumulative_revenue
    print(f"\nAdditional Revenue from Trust Investment: {additional_revenue:,.0f} yen")
    print(f"Average Monthly Uplift: {additional_revenue/60:,.0f} yen/month")

    # 複利ループの説明
    print("\n## Compound Loop Structure")
    print("-"*60)
    print("""
    Trust Quality Investment (5 → 8)
           │
           ├──→ Repurchase Rate ↑ (+4.5pt) ──→ Higher Retention
           │                                        │
           ├──→ Churn Rate ↓ (-1.5pt) ────────→ Larger Customer Base
           │                                        │
           ├──→ UGC Rate ↑ (+2.4pt) ──→ Trust Signal Accumulation
           │                                   │
           └──→ Referral Rate ↑ (+3pt) ────────┤
                                               │
                                               ↓
                                    CVR Improvement (Trust Effect)
                                               │
                                               ↓
                                    More New Customers
                                               │
                                               ↓
                                    Larger Base × Higher Retention
                                               │
                                               └──→ COMPOUND LOOP

    Key Insight: The compound effect accelerates over time because:
    1. UGC accumulates (trust signal grows)
    2. CVR improves → More customers acquired per visitor
    3. More customers × Higher retention = More purchases
    4. More purchases × Higher UGC rate = More trust signal
    5. Loop repeats with ever-growing base
    """)

    print("="*80)


def main():
    """メイン実行"""

    print("EEZO Trust-Based Compound Growth Simulator (Rigorous Version)")
    print("-"*60)
    print("Simulating 5-year impact of trust quality investment...")

    # シミュレーション実行
    baseline, invested = run_comparison(months=60)

    # グラフ出力
    plot_results(baseline, invested)

    # 詳細レポート
    baseline_params = SimulationParams(trust_quality=5.0, months=60)
    invested_params = SimulationParams(trust_quality=8.0, months=60)
    print_detailed_report(baseline, invested, baseline_params, invested_params)


if __name__ == "__main__":
    main()
