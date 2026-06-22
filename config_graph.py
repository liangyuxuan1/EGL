from typing import Literal, Optional, Dict, Any
from dataclasses import dataclass, field

Stage = Literal["A", "B"]   # 参数设置阶段 A/B

@dataclass
class GraphUpdateConfig:
    """
    Graph-aware REINFORCE + Gibbs sampling update config with staged scheduling.

    Usage pattern (your desired style):
      - KnowledgeGraph has: self.update_cfg = GraphUpdateConfig()
      - In KG methods: use self.update_cfg.lr_theta, self.update_cfg.bias_lr_p, ...
      - In main loop: only call global_kg.update_cfg.change_stage("B") at some iteration.
    """

    # -------------------------
    # (0) stage controller
    # -------------------------
    stage: Stage = "A"   # current active stage ("A" or "B")

    # -------------------------
    # (1) REINFORCE / theta-phi update (base knobs)
    # -------------------------
    lr_theta_A: float = 0.03    # 0.05 偏大；0.01~0.03 更稳（减少长期漂移）
    lr_phi_A: float = 0.02      # 0.02 偏激进；phi 边多且噪声大，建议更小

    lr_theta_B: float = 0.015   # stage B 减半. 不建议大幅改 lr_theta/lr_phi（你验证过对 flip_rate 没啥用）
    lr_phi_B: float = 0.01

    adv_clip: float = 2.5
    ent_coef: float = 0.0       # per your request: no entropy
    ema_baseline: float = 0.9   # baseline EMA

    # weight decay (light L2-like shrink)
    theta_decay_A: float = 0.0005   # 每步 theta <- (1-lam)*theta，建议比 phi 更小一点， 经验：2e-4 ~ 2e-3；先用 5e-4 控制 std_theta 单调涨
    phi_decay_A: float = 0.0005      # e.g., 1e-3 ~ 1e-2， 每步 phi <- (1-lam)*phi

    theta_decay_B: float = 0.001    # stage B 加大一倍
    phi_decay_B: float = 0.001

    # trust gating speed (support -> trust weight)
    # trust_weight = 1 -  exp(-support / tau)
    # edge support上升的速度比node要慢得多，所以edge的tau要小于node的tau
    # 避免confirm被node主导
    tau_node: float = 20.0          # tau_node = 20 -> s=10: tw = 1-exp(-0.5) = 0.393, s=20: tw = 1-exp(-1) = 0.632, s=40: tw = 0.865
    tau_edge: float = 4.0           # tau_edge = 4 -> s=1: tw = 0.221, s=3: tw = 0.528, s=5: tw = 0.713
    trust_min_w_edge_A: float = 0.05  # keep tiny weight so early stage isn't "phi-dead"
    trust_min_w_edge_B: float = 0.03  # keep tiny weight so early stage isn't "phi-dead"
    trust_min_w_node: float = 0.02  # optional symmetry

    # -------------------------
    # (2) Gibbs sampling knobs
    # -------------------------
    # 温度控制graph本身证据场的影响，早期证据不足，温度设大，让LLM驱动搜索
    # 后期降低温度， 让证据场本身驱动。LLM的驱动与温度无关
    temperature_A: float = 1.2     # higher -> more random; lower -> more peaked
    temperature_B: float = 0.7

    # gibbs_T 是一轮中更新多少次（扫多少遍），扫得越多越接近该温度下的平衡分布（反而可能更“跳”）
    # 前期：T 稍大 + gibbs_T 稍大（混合快，探索）
    # 后期：T 变小 + gibbs_T 变小（更粘连，收敛）
    gibbs_T_A: int = 2            # sweeps
    gibbs_T_B: int = 1            # more sweeps later helps mixing under stronger couplings

    # -------------------------
    # (3) bias0 PI controller knobs (keep k in band)
    # -------------------------
    # 控制期望的 k 值，bias0 = log （p0 / (1-p0)）
    # 如果期望k0 = 35, 则 bias0 = log (35/75 / (1-35/75)) = -0.133
    # 如果期望k0 = 30，则 bias0 = log (0.4 / (1-0.4)) = -0.405
    # k0 = 20， bias0 = -1.01
    # k0 = 10, bias0 = -1.87, 5: -2.64, 3: -3.178, 2: -3.59, 74: 4.3, 50: 0.69, 40: 0.135
    bias0: float = -1.47  # k 12: -1.66, 14: -1.47

    bias_k_low_A: int = 12
    bias_k_high_A: int = 16
    bias_lr_p_A: float = 0.10
    bias_lr_i_A: float = 0.01

    bias_k_low_B: int = 14
    bias_k_high_B: int = 20
    bias_lr_p_B: float = 0.06
    bias_lr_i_B: float = 0.005

    bias_ema_beta: float = 0.92
    bias_clamp_min: float = -3.2    # 最小k=3~4
    bias_clamp_max: float = -0.9    # 最大k<20

    edge_size: int = 1000  # candidate sparse edges, 目前数据规模不使用该参数

    # ============================================================
    # Below are the "active" fields used by KnowledgeGraph methods.
    # They get overwritten when stage changes.
    # ============================================================

    # active reinforce
    lr_theta: float = field(init=False)
    lr_phi: float = field(init=False)
    theta_decay: float = field(init=False)
    phi_decay: float = field(init=False)

    # active sampler
    temperature: float = field(init=False)
    gibbs_T: int = field(init=False)

    # active bias controller
    bias_k_low: int = field(init=False)
    bias_k_high: int = field(init=False)
    bias_lr_p: float = field(init=False)
    bias_lr_i: float = field(init=False)

    trust_min_w_edge: float = field(init=False)

    # NEW: transition controls (optional defaults)
    transition_len: int = 100      # default warm transition length
    transition_start_iter: int = 200  # you can set in main loop if needed

    # -------------------------
    # LLM agenda odds multipliers (sampling shaping)
    # -------------------------
    lambda_must_A: float = 5.0
    lambda_prefer_A: float = 3.0
    lambda_avoid_A: float = 3.5
    lambda_frontier_endpoint_A: float = 3.0

    lambda_must_B: float = 2.5
    lambda_prefer_B: float = 1.6
    lambda_avoid_B: float = 1.8
    lambda_frontier_endpoint_B: float = 1.6

    # active (materialized)
    lambda_must: float = field(init=False)
    lambda_prefer: float = field(init=False)
    lambda_avoid: float = field(init=False)
    lambda_frontier_endpoint: float = field(init=False)

    def __post_init__(self) -> None:
        # initialize active params for the default stage
        self.change_stage(self.stage)

    # -------------------------
    # NEW helpers
    # -------------------------
    def _stage_values(self, stage: Stage) -> Dict[str, Any]:
        """Return the target values for a given stage (A or B) WITHOUT side-effects."""
        stage = "A" if stage not in ("A", "B") else stage

        if stage == "A":
            return {
                "lr_theta": float(self.lr_theta_A),
                "lr_phi": float(self.lr_phi_A),
                "theta_decay": float(self.theta_decay_A),
                "phi_decay": float(self.phi_decay_A),
                "temperature": float(self.temperature_A),
                "gibbs_T": int(self.gibbs_T_A),
                "bias_k_low": int(self.bias_k_low_A),
                "bias_k_high": int(self.bias_k_high_A),
                "bias_lr_p": float(self.bias_lr_p_A),
                "bias_lr_i": float(self.bias_lr_i_A),
                # keep your intended A-boost
                "trust_min_w_edge": float(self.trust_min_w_edge_A),
                "lambda_must": float(self.lambda_must_A),
                "lambda_prefer": float(self.lambda_prefer_A),
                "lambda_avoid": float(self.lambda_avoid_A),
                "lambda_frontier_endpoint": float(self.lambda_frontier_endpoint_A),
            }
        else:
            return {
                "lr_theta": float(self.lr_theta_B),
                "lr_phi": float(self.lr_phi_B),
                "theta_decay": float(self.theta_decay_B),
                "phi_decay": float(self.phi_decay_B),
                "temperature": float(self.temperature_B),
                "gibbs_T": int(self.gibbs_T_B),
                "bias_k_low": int(self.bias_k_low_B),
                "bias_k_high": int(self.bias_k_high_B),
                "bias_lr_p": float(self.bias_lr_p_B),
                "bias_lr_i": float(self.bias_lr_i_B),
                "trust_min_w_edge": float(self.trust_min_w_edge_B),
                "lambda_must": float(self.lambda_must_B),
                "lambda_prefer": float(self.lambda_prefer_B),
                "lambda_avoid": float(self.lambda_avoid_B),
                "lambda_frontier_endpoint": float(self.lambda_frontier_endpoint_B),
            }

    @staticmethod
    def _blend(a: float, b: float, s: float) -> float:
        return (1.0 - s) * float(a) + s * float(b)

    @staticmethod
    def _smoothstep(s: float) -> float:
        """C^1 smoothstep for nicer transition than linear."""
        s = max(0.0, min(1.0, float(s)))
        return s * s * (3.0 - 2.0 * s)

    # -------------------------
    # UPDATED: change_stage with optional soft transition
    # -------------------------
    def change_stage(
        self,
        stage: Stage,
        *,
        iteration: Optional[int] = None,
        transition_len: Optional[int] = None,
        transition_start_iter: Optional[int] = None,
        use_smoothstep: bool = True,
    ) -> None:
        """
        Switch active stage and materialize "active" fields.

        If iteration is provided, we do a soft transition from A -> B over `transition_len`
        iterations starting at `transition_start_iter`.

        Typical usage in main loop:
            if it == t0:   cfg.change_stage("B", iteration=it, transition_start_iter=t0, transition_len=50)
            else:          cfg.change_stage("B", iteration=it, transition_start_iter=t0, transition_len=50)
        """

        stage = "A" if stage not in ("A", "B") else stage
        self.stage = stage

        # default transition params
        if transition_len is None:
            transition_len = int(getattr(self, "transition_len", 0) or 0)
        if transition_start_iter is None:
            transition_start_iter = int(getattr(self, "transition_start_iter", 0) or 0)

        # If no iteration info, just do hard assignment
        if stage != "B" or iteration is None or transition_len <= 0:
            vals = self._stage_values(stage)
            self.lr_theta = vals["lr_theta"]
            self.lr_phi = vals["lr_phi"]
            self.theta_decay = vals["theta_decay"]
            self.phi_decay = vals["phi_decay"]
            self.temperature = vals["temperature"]
            self.gibbs_T = vals["gibbs_T"]
            self.bias_k_low = vals["bias_k_low"]
            self.bias_k_high = vals["bias_k_high"]
            self.bias_lr_p = vals["bias_lr_p"]
            self.bias_lr_i = vals["bias_lr_i"]
            self.trust_min_w_edge = vals["trust_min_w_edge"]
            self.lambda_must = vals["lambda_must"]
            self.lambda_prefer = vals["lambda_prefer"]
            self.lambda_avoid = vals["lambda_avoid"]
            self.lambda_frontier_endpoint = vals["lambda_frontier_endpoint"]
            return

        # Soft transition only for A -> B
        A = self._stage_values("A")
        B = self._stage_values("B")

        # progress s in [0,1]
        t0 = int(transition_start_iter)
        t1 = int(transition_start_iter) + int(transition_len)
        if iteration <= t0:
            s = 0.0
        elif iteration >= t1:
            s = 1.0
        else:
            s = (float(iteration) - float(t0)) / max(1.0, float(transition_len))

        if use_smoothstep:
            s = self._smoothstep(s)

        # Blend continuous params
        self.lr_theta = self._blend(A["lr_theta"], B["lr_theta"], s)
        self.lr_phi = self._blend(A["lr_phi"], B["lr_phi"], s)
        self.theta_decay = self._blend(A["theta_decay"], B["theta_decay"], s)
        self.phi_decay = self._blend(A["phi_decay"], B["phi_decay"], s)
        self.temperature = self._blend(A["temperature"], B["temperature"], s)
        self.bias_lr_p = self._blend(A["bias_lr_p"], B["bias_lr_p"], s)
        self.bias_lr_i = self._blend(A["bias_lr_i"], B["bias_lr_i"], s)
        self.trust_min_w_edge = self._blend(A["trust_min_w_edge"], B["trust_min_w_edge"], s)

        self.lambda_must = self._blend(A["lambda_must"], B["lambda_must"], s)
        self.lambda_prefer = self._blend(A["lambda_prefer"], B["lambda_prefer"], s)
        self.lambda_avoid = self._blend(A["lambda_avoid"], B["lambda_avoid"], s)
        self.lambda_frontier_endpoint = self._blend(A["lambda_frontier_endpoint"], B["lambda_frontier_endpoint"], s)

        # Blend integer/band params: keep them discrete, switch near the end
        self.bias_k_low = int(round(self._blend(A["bias_k_low"], B["bias_k_low"], s)))
        self.bias_k_high = int(round(self._blend(A["bias_k_high"], B["bias_k_high"], s)))

        # gibbs_T: switch late to avoid abrupt mixing change at the same time as temperature
        self.gibbs_T = int(A["gibbs_T"] if s < 0.8 else B["gibbs_T"])