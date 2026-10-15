> Historical design notes. Current supported HP, output schemas and analysis
> contracts are documented in [PROBING.md](application/EMT_disconnected_1pt/PROBING.md).
> Old alternatives below are not compatibility promises.

# Disconnected EMT：HP 原理、现有问题、修正算法与 Multiplier 开发方案

状态（2026-09-12）：已实现 CPU 数学测试、EMT HP 修复、固定 plan 的 Multiplier 接入及完整 base 聚合；真实 HIP/MPI 验证仍待执行。第 3 节保留修复前审计背景，实施结果与命令见第 11 节。不修改原始 `software/Pyquda_Measurement`。

HP 范围要求：本计划仅讨论并测试 HP2、HP32、HP512 三个完整层。每个 base noise 必须完成所选层的全部 vectors；其他 HP 数量不作为本轮运行选项、对照组或性能曲线。此限制针对 HP，不改变普通 noise 和 multiplier 的测试方案。

本文按一个顺序回答问题：我们在估计什么 → probing 为什么能降噪 → HP 应该做什么 → 当前程序实际做了什么 → 如何修正 → multiplier 是什么及是否值得加入 → 如何实施和验证。

文中的“代码事实”来自此前对本地实现的检查；“推导”给出可独立验证的数学依据；“建议”是针对本项目的工程判断。候选 plan 的已有检查不能替代后续可复现测试。文献及其适用范围集中列在第 9 节。

## 1. 我们要解决什么问题

### 1.1 Disconnected loop 为什么需要随机源

固定一份 gauge configuration 后，disconnected loop 的核心可以写成某个大矩阵的 trace：

\[
T=\operatorname{Tr}A.
\]

这里的矩阵指标不仅包含四维格点，也包含 spin 和 color。最简单的例子是包含 Dirac 逆矩阵的算子；实际 EMT 还涉及导数、gamma 矩阵、flow 和投影。矩阵太大，不能逐列求逆后求 trace，因此使用随机估计。

取独立复相位噪声 \(\xi\)，满足 \(E[\xi_a^*\xi_b]=\delta_{ab}\)，则：

\[
\widehat T=\xi^\dagger A\xi
=\sum_a A_{aa}+\sum_{a\ne b}\xi_a^*A_{ab}\xi_b,
\qquad E[\widehat T]=T.
\]

对于单位模噪声，对角部分在每个样本中已经正确；误差来自非对角项的随机残留。对独立 Z4 噪声，以复方差 \(E[|\widehat T-T|^2]\) 定义误差时：

\[
\operatorname{Var}(\widehat T)=\sum_{a\ne b}|A_{ab}|^2.
\]

这解释了 probing 的目标：不是改变 Dirac 方程，也不是修改 EMT 算子，而是通过设计 source，消掉一部分非对角随机项。上述方差公式不能不加区分地用于 Z2 噪声、实部/虚部分量或非线性后处理。

### 1.2 为什么优先消掉短距离项

若重要的非对角矩阵元集中在空间上接近的点之间，优先消去这些项就可能显著降噪。这是利用格子几何设计 probing 的动机，也是 HP 文献的出发点。[R1]

但对本项目，不能直接把裸 \(D^{-1}\) 的局域性当作所有 EMT observable 的方差模型。例如固定 gauge 下，若 noise 和 solution 都经过线性 flow \(F_\tau\)，bilinear insertion 为 \(B\)，则有效估计可包含：

\[
\xi^\dagger F_\tau^\dagger B F_\tau D^{-1}\xi.
\]

这只是解释噪声结构的简化形式，实际左右导数、gamma5 对称化和投影必须保留原定义。flow time、kinetic 和 T44 的最优 probing 不一定相同；带位移 observable 需要专门考虑其邻域，也有文献先例。[R4]

### 1.3 本轮开发的目标和边界

目标是在相同物理参数、solver 容差和 flow 定义下，比较普通 Z_N noise、修正后的 HP、multiplier 的统计效率。先验证算法，再用实际 EMT 判断哪种方法更省计算时间。

本轮只讨论 source/probing、必要的结果聚合和测试，不同时引入低模、spin/color dilution 或修改 EMT kernel。当前完整 spin/color noise 的一个 probing vector 对应一次 fermion inversion，不是 12 次 dilution inversions。site-only probing 也不会消去同一 site 内不同 spin/color 的所有噪声。

## 2. Probing 与 HP 的原理：它们究竟做了什么

### 2.1 从一个随机源构造一组相关的源

对同一个 base noise \(\xi\)，乘上确定性的单位模 pattern：

\[
\eta_k=P_k\xi,\qquad (P_k)_{aa}=p_k(a),\qquad |p_k(a)|=1.
\]

使用 m 个 vectors 的平均估计为：

\[
\widehat T_m=\frac1m\sum_{k=0}^{m-1}\eta_k^\dagger A\eta_k
=\sum_{a,b}\xi_a^*A_{ab}\xi_b K_m(a,b),
\quad K_m(a,b)=\frac1m\sum_k p_k(a)^*p_k(b).
\]

\(K_m\) 是理解整个算法的关键：对某个 pair，若 \(K_m(a,b)=0\)，该非对角项对这个 base 的贡献就被精确消掉，不需要等待随机平均。

所有 vectors 必须共用同一个 base noise，才能发生这种逐项抵消。每个 vector 换一个独立 noise 仍可无偏，但不再拥有上述 probing 抵消结构。

### 2.2 着色如何变成消去规则

先给每个 site 一个颜色 \(c(x)\)。如果完整的一组 patterns 满足：

\[
K(x,y)=\delta_{c(x),c(y)},
\]

则不同颜色的 pair 被消去，只剩同色 pair。这里的“颜色”是格点图的标签，不是 QCD 的 SU(3) color。

在周期四维格子上定义图距离：

\[
d_G(x,y)=\sum_{\mu=1}^4\min(|x_\mu-y_\mu|,L_\mu-|x_\mu-y_\mu|).
\]

distance-d coloring 的要求是：距离在 1 到 d 之间的任意两个点不能同色。完成对应 basis 后，这些短距离 pair 就全部被消去。

### 2.3 HP 比一般着色多做了什么

HP 的核心不是“使用 ±1”，而是把着色组织成嵌套层次，再安排 Hadamard/Walsh 向量的顺序，使已计算向量可以成为更细层的前缀。这是其相对独立 distance-coloring 的主要结构优势。[R1]

可以把它理解为：先区分最近邻，再继续细分颜色，让更远的点也能被区分；细分时不废弃之前的 vectors。

若 site 的二进制标签为 \(\ell(x)\)，Walsh pattern 可写为：

\[
h_k(x)=(-1)^{\operatorname{popcount}(k\mathbin{\&}\ell(x))}.
\]

当使用前 \(N=2^m\) 个整数 masks 时：

\[
K_N(x,y)=\prod_{j=0}^{m-1}\frac{1+(-1)^{\ell_j(x)\oplus\ell_j(y)}}2.
\]

因此，前 N 个 vectors 实际按标签的最低 m bits 着色。要获得正确的空间距离保证，就必须让这些最低 bits 表示正确的几何分组。Walsh 正交性本身不会替我们完成这件事。

### 2.4 三个必须区分的概念

1. **无偏性**：固定单位模 patterns、归一化正确时，有 \(E[\widehat T_m]=T\)；这与距离消去保证是不同条件。本计划的 HP 一律完成所选层后统计。
2. **距离消去保证**：必须检查 \(K_m\) 对目标短距离 pair 是否为零；仅有无偏性不够。
3. **统计效率**：消去一些 pair 不自动意味着同等 wall time 下最优，还取决于剩余方差与计算成本。

完整 basis 内的 vectors 是相关样本。独立统计单位应是“一个独立 base noise 及其整组 probing 测量”，不能把全部 vectors 当成独立 Monte Carlo samples。

## 3. 修复前程序做了什么，问题在哪里

本节描述实施前状态，用于解释修复动机，不代表修复后的接口现状。

### 3.1 已有的计算流程

相关入口是：

- `pyquda_measurement_utils/Disconnected_utils_vibe_develop.py`：noise、HP pattern、参数检查和 resume 身份。
- `pyquda_measurement_utils/Disconnected_1pt_EMT_vibe_develop.py`：source 迭代、反演、flow、收缩和 shards。
- `application/EMT_disconnected_1pt/`：应用调用入口。

此前代码检查确认：同一个 HP base 重用确定性 counter noise，跨 batch 可以重建同一 source；pattern 会转换到对应 backend；反演使用 probed source，收缩包含对应 noise 的复共轭。这些机制应保留，不应因修正 ordering 而重写。

本地 PyQUDA 的 `coordinate()` 源码返回包含 MPI offsets 的全局 XYZT 坐标，按 even-odd layout 排列；真实 GPU/MPI 的 source 一致性仍应加入测试。

### 3.2 修复前 HP 的标签与向量顺序

默认 `hp_ordering="interleaved_xyzt_binary_projected_to_evenodd"` 把坐标 bits 直接交织为：

```text
x0, y0, z0, t0, x1, y1, z1, t1, ...
```

然后以自然整数 `hp_idx` 作为 Walsh mask。`hierarchical_gray_index()` 的名称不能说明默认分支已经进行了合适的 HP 排序；关键仍是前缀实际使用哪些 bits。

### 3.3 为什么这不满足预期的 HP 距离层次

第 2 节的公式说明，旧 HP2 只看 x0。因此第二个 pattern 是 x 方向奇偶性，而不是四维 checkerboard；沿 y、z、t 的最近邻可能仍然同色。

同样，旧 HP32 看四个最低坐标 bits 和 x1。这种细分偏向单一方向，不能直接解释为完整四维 distance-2 层。

下表中的反例均可在 8^4 上直接重建：

| 旧前缀 | 同色点对 | 周期距离 | 说明 |
|---|---|---:|---|
| HP2 | 原点与 (0,1,0,0) | 1 | y 最近邻未消去 |
| HP32 | 原点与 (0,2,0,0) | 2 | 加入 x1 不能区分 y 方向距离 2 |
| HP512 | 原点与 (0,4,0,0) | 4 | 更高层仍有同类方向问题 |

**判断：需要修正。** 问题是现有命名对应的预期几何消去不成立，不是已经证明旧物理均值有偏。旧结果应保留并标注 legacy ordering；不能把它们继续当作正规 HP2/32/512 的性能证据。仅换 Gray code 也不是充分修复，必须验证空间 pair 的消去。

### 3.4 与 ordering 分开处理的流程限制

修复前仅支持 `zn` 和 `hierarchical_probing`。`_measure_base_shards()` 把非 HP 情况作为单 source 的普通噪声处理；仅增加一个 multiplier 名称会有只运行 k=0 的风险。

此外，默认只保存 shard means，而 raw finalizer 不接受 mean-only shards；修复前没有完整的 base-level SEM 分析路径。resume 以完整 base 为单位，fingerprint 包含 vector 数和 ordering。新增的 base 聚合接口解决 mean-only 分析，但“数学上嵌套”仍不等于已支持 HP 跨层续接。

这些是后续接入和分析必须处理的接口问题，应与 HP 的数学错误区分，不应一概描述为现有均值计算错误。

## 4. HP 修正方案：新的标签、新的算法及其保证

### 4.1 设计思路：先四维 checkerboard，再逐尺度细分

最低标签 bit 改为：

\[
b_0=x_0\oplus y_0\oplus z_0\oplus t_0.
\]

这样 HP2 的两个 patterns 就是全 1 和四维 checkerboard，相邻点在第二个 pattern 上符号相反，平均后最近邻项恰好抵消。

升级到下一尺度时，追加 x0、y0、z0，以及下一尺度的总 parity。已有 b0 加上前三个坐标 bit 足以恢复 t0，所以无须重复保存它：

```text
HP2:   parity(x0,y0,z0,t0)
HP32:  append x0, y0, z0, parity(x1,y1,z1,t1)
HP512: append x1, y1, z1, parity(x2,y2,z2,t2)
```

最终仍使用同一个 Walsh 公式，只把旧交织标签换成这个 nested label。新算法的实质是修正“空间标签到前缀”的映射，而不是换掉反演或 EMT 测量。

### 4.2 可重建的核心算法

以下为参考伪代码，`coords` 必须是具有相同本地 shape 的全局 XYZT 坐标。正式实现还需检查参数、整数位宽和 backend；这里不以文档代码代替测试。

```python
def nested_label(coords, max_scale):
    label = xor_all(coord & 1 for coord in coords)
    out_bit = 1
    for r in range(max_scale):
        for coord in coords[:3]:
            label |= ((coord >> r) & 1) << out_bit
            out_bit += 1
        parity = xor_all((coord >> (r + 1)) & 1 for coord in coords)
        label |= parity << out_bit
        out_bit += 1
    return label

def hp_pattern(label, probe_index):
    return 1 - 2 * (popcount(label & probe_index) % 2)

for base_index in range(num_bases):
    base_noise = make_noise(configuration, seed, base_index)
    for probe_index in range(vectors_per_base):
        pattern = to_backend(hp_pattern(label, probe_index))
        source = base_noise.copy()
        source.data *= pattern[..., None, None]
        measure(source)
```

`max_scale=0/1/2` 分别足以生成 HP2/32/512；生成更高层标签不会改变低层 patterns。`xor_all`、`popcount` 表示逐元素操作，不是 Python 标量函数。

### 4.3 为什么完整层是 2、32、512，以及距离保证

本计划取尺度 \(q\in\{1,2,4\}\)，且每个 \(L_\mu\) 可被 \(2q\) 整除。完整层的标签等价于记录：每个坐标模 q 的余数，以及粗坐标 \(\lfloor x_\mu/q\rfloor\) 的总 parity。

四个方向各有 q 种余数，组合数为 \(q^4\)；每种组合再按粗 checkerboard 分成两类，因此：

\[
\boxed{N_{\mathrm{colors}}=q^4\times2}
\]

使用这一颜色空间的完整 Walsh 基时，\(N_p=N_{\mathrm{colors}}\)。标签 bits 数为 \(m=4\log_2q+1\)，完整基有 \(2^m\) 个 vectors。于是 q=1、2、4 分别对应 HP2、HP32、HP512；尺度加倍时四个方向的余数信息各增加一个 bit，向量数乘以 16。这里“完整”指所选层颜色空间上的完整基，不是整个 lattice 自由度的完整基。

同色点的各方向周期距离必须是 q 的倍数。若两个不同点的总距离小于 2q，只能有一个方向相差 q；但这会翻转粗 parity，与同色矛盾。跨周期边界时 \(L_\mu/q\) 为偶数，因此证明仍成立。

所以同色点的最短距离至少为 2q，所有距离不超过 \(2q-1\) 的 pair 被消去。在两个方向各移动 q 可达到同色距离 2q，故在这里讨论的四维格子上界限是可达到的。

| 完整前缀 | q | 标签 bits | Vectors/base | 保证消去至距离 |
|---|---:|---:|---:|---:|
| HP2 | 1 | 1 | 2 | 1 |
| HP32 | 2 | 5 | 32 | 3 |
| HP512 | 4 | 9 | 512 | 7 |

这里 \(N_p=2q^4\) 是这套构造的数量，不是一般图着色最少颜色数的定理。本轮三个完整层用于目标 distance 1、2、4，同时报告实际消去半径 1、3、7。

上述 bit 构造及证明是本计划的独立推导；它遵循 HP 的嵌套着色思想，不声称逐行复现某篇论文的实现。

### 4.4 完整层运行要求

HP 的运行参数必须限定为 `vectors_per_base in {2, 32, 512}`，并验证对应的 shape 条件。其他数量应在运行前明确拒绝，不静默取整或降级。

每个 base 共用同一 noise，遍历所选层的全部 vectors，再生成该 base 的最终估计和统计记录。batch/shard 只是执行与存储切分，不是新的 HP 层；未完成整层的 base 不参与最终均值、SEM 或效率排名。

三个完整层分别验证、分别报告。数学上的嵌套可为将来复用提供基础，但本轮不把扩层续跑作为必需功能，也不讨论其他数量的 HP 性能。

### 4.5 实现边界与兼容性

初版限定四维、每个 extent 至少为 2 且为 2 的幂，要求 \(q\le\min L_\mu/2\)。这是保守的实现范围；证明本身可支持部分非二次幂格子的特定尺度，但不能据此无测试放开所有 shape。任意边长的 HP 扩展可参考 [R3]。

建议新增 `nested_xyzt_checkerboard_v1`，保留旧 ordering。ordering、算法版本和几何必须进入 metadata 与 resume fingerprint。负索引、越界索引、非整数输入和位宽均需检查；index=0 的快捷返回也不能绕过验证。

只调整 label/pattern 路径，保留已正确的 counter noise、backend、反演、flow 和收缩。我的判断是先做好固定 Np 的续跑；跨 HP 层复用另行实现，避免把数学修复与复杂状态迁移绑定。

## 5. Multiplier：另一种着色算法，而不是 HP 的新名字

### 5.1 它做了什么

Multiplier 直接尝试一个线性模着色：

\[
c(x)=\sum_\mu\sigma_\mu x_\mu\pmod{n_c}.
\]

\(n_c\) 是颜色数，\(\sigma\) 是按 XYZT 排列的整数系数。搜索目标是找到满足目标 distance 的合法着色，而不要求 M1、M2、M4 相互嵌套。M2 的 2 是目标距离，不是两个 vectors。

获得合法颜色后，使用完整 Fourier basis：

\[
p_k(x)=e^{2\pi i k c(x)/n_c},\quad k=0,\ldots,n_c-1,
\qquad \frac1{n_c}\sum_kp_k(x)^*p_k(y)=\delta_{c(x),c(y)}.
\]

所以它与 HP 一样，通过共享 base noise 和完整 basis 消去异色 pair；区别在于颜色的构造方式和 basis。HP 使用二进制标签上的 Walsh characters，multiplier 使用循环模群的 Fourier characters。

**有效性判断：** 只要周期着色合法、完整 basis 和归一化正确，这个消去机制有直接的代数证明。需要验证的是具体 plan 与工程实现，不是通过有限样本“猜测”正交性是否成立。

### 5.2 周期边界是最重要的正确性条件

不能只检验短整数位移 \(\sigma\cdot\Delta\ne0\pmod{n_c}\)。实际跨界点的颜色差是：

\[
c((x+\Delta)\bmod L)-c(x)
=\sigma\cdot\Delta-\sum_\mu\sigma_\mu L_\mu w_\mu\pmod{n_c}.
\]

只有每个 \(\sigma_\mu L_\mu\) 都是 \(n_c\) 的倍数时，才可无条件省略 wrap 项。这是简化验证的充分条件，不是所有合法线性着色必须满足的条件。

反例：8^4、9 色、\(\sigma=(1,2,3,4)\)。点 (1,0,0,0) 与 (0,0,0,7) 的周期距离为 2，颜色却同为 1。只检查原点附近短 signed displacements 会漏掉这类跨界冲突。

可靠 checker 应遍历所有目标短位移及各方向可实现的 direct/wrapped 坐标差组合；8^4 再以实际点对独立交叉验证。搜索器与 checker 不应共享未经验证的剪枝。fermion 反周期边界改变耦合符号，不会把跨界最近邻从图中删除。

### 5.3 候选 plans 与“10 色”未决问题

此前独立检查得到以下候选；实现阶段仍须把检查固化成可复现测试，再允许用于运行：

| Shape | 目标 d | 颜色数 / vectors | sigma |
|---|---:|---:|---|
| 8^4 | 1 | 2 | (1,1,1,1) |
| 8^4 | 2 | 16 | (1,2,3,4) |
| 64^4 | 4 | 64 | (1,4,10,17) |

M1 与新 HP2 就是同一 checkerboard；相同 base 下，它们应给出数值精度内相同的结果。这是首个集成测试。

已有枚举记录在严格周期、线性模着色的定义下，对 8^4 和 64×32^3 的 distance-2 未找到 9–15 色方案，16 色有解。少于 9 色可由中心和八个最近邻构成的 distance-2 clique 排除。但“未找到”的记录应在保存完整枚举程序、搜索范围和验证输出后，才能作为可审计的穷举结论；也不能外推为一般图着色下界。

Papace 等预印本的 Table 1 列出了 64×32^3、distance-2、10 色结果，相关搜索文献尚标为准备中。[R2] 这与上述检查存在未解决的冲突。需要取得具体 sigma、坐标定义及周期处理来复现，不能仅凭表格接受不通过 checker 的 plan，也不应在未复现前断言论文整体错误。

### 5.4 如何生成、保存和使用 plan

初版先使用固定候选并独立验证，不把昂贵组合搜索放进 MPI 生产任务。后续离线搜索按颜色数递增，在模 nc 的系数空间寻找方案；只有保持 shape 的方向置换才能用于对称性消重，剪枝需有正确性依据。

plan 应保存 shape、目标 distance、nc、XYZT sigma、算法版本和验证定义。运行前统一读取，各 rank 使用同一 plan。改变 shape 或 distance 后必须重新验证，不直接跨格子搬用。

源的迭代与 HP 相同：每个 base 生成一次 noise，然后遍历全部 nc 个 Fourier patterns。必须显式新增 multiplier 分支、vectors_per_base 和完整 probe 区间；仅增加 pattern 函数或允许一个新 scheme 名称是不够的。

### 5.5 Fourier 与颜色 mask 的归一化不能混用

令 \(Q_c\) 为颜色 c 的 site projector。两种表示在同一个 noise、完整 basis 和精确线性求解下满足：

\[
\underbrace{\sum_c(Q_c\xi)^\dagger A(Q_c\xi)}_{\text{unscaled color-mask sum}}
=\underbrace{\frac1{n_c}\sum_k(P_k\xi)^\dagger A(P_k\xi)}_{\text{Fourier average}}.
\]

未缩放 0/1 mask 是求和，不除颜色数；单位模 Fourier 是平均，需要除颜色数。文献 [R2] 的颜色 mask 约定不能直接代入本项目的 source-average 接口。

建议初版采用 Fourier，因其与现有单位模 source 及均值约定更接近。用小矩阵测试上述逐 base 恒等式；实际 solver 下允许容差控制的误差。将来若比较两种 source 表示的 solver 成本，再独立加入 mask。

## 6. Multiplier 是否有优势，我是否建议加入

### 6.1 有成本潜力，但不能把颜色数比值当成加速比

| 对照 | M vectors/base | HP vectors/base | 实际消去范围的区别 |
|---|---:|---:|---|
| M1 / HP2 | 2 | 2 | 都至距离 1 |
| M2 / HP32 | 16 | 32 | 所列 M2 至 2，新 HP32 至 3 |
| M4 / HP512 | 64 | 512 | 所列 M4 至 4，新 HP512 至 7 |

后两行是匹配最低目标距离的对照，不是完全相同的消去范围。所列 M2 有同色差向量 (1,1,−1,0)，距离 3；M4 有 (1,−2,−1,1)，距离 5，因此它们的最大保证分别到 2、4。

Multiplier 可能用更少反演、flow 和收缩完成一个 base，但也可能保留更多重要残余项。HP 的额外 vectors 不一定浪费；应评价其额外消去是否值得。不同 multiplier plans 通常不嵌套，也不能默认复用低 distance 的所有 solves。

### 6.2 正确的效率指标

令完整 base 的方差为 V，成本为 C，总预算为 B。在相同运行布局、固定方法和独立 bases 下：

\[
\operatorname{Var}(\overline T)\simeq\frac{VC}{B},\qquad
R=\frac{V_M C_M}{V_{HP}C_{HP}}.
\]

R<1 才表示 multiplier 更省预算。若每 vector 成本近似相同，M2 对 HP32 获胜需要 \(V_M/V_{HP}<2\)，M4 对 HP512 需要该比值小于 8。这是可检验的条件，不是预测结果。

要同时报告相同 solves 和相同 wall time 下的误差，以及包含初始化/warm-up 的实际成本和稳态成本。不同节点布局还应比较 node-hours。kinetic、T44、不同 flow times 分别评价；中心值接近零时不使用不稳定的相对误差。

### 6.3 我的建议

建议加入 multiplier，但作为经过验证的实验选项，不直接替代 HP 生产默认值。先修正 HP，才能获得可信对照；再接入固定 multiplier plans，用小格子和有限预算 pilot 决定是否扩大运行。

第一轮不建议同时做复杂 sigma 搜索、variance-aware 排名或任意 HP 层续接。若 pilot 确认有收益，再投入这些工程工作。只有颜色更少、某条曲线更平滑或单次 SEM 更低，都不足以证明长期优势。

## 7. 两种新方法共用的工程与统计要求

### 7.1 按 base 聚合，不按相关 vectors 估计 SEM

\[
T_b=\frac1{N_p}\sum_kT_{b,k},\qquad
\overline T=\frac1{N_b}\sum_bT_b.
\]

对实部或虚部分量，可由独立 \(T_b\) 的样本方差除以 Nb 得到均值方差；同一 base 内的所有 vectors 不能当独立样本。多个 flow times 共用 noise，应保留其 base 级联合数据和相关性。

不同长度 shards 必须按实际 count 合并。例如 16 vectors 分成 6+6+4，base mean 应为 \((6m_0+6m_1+4m_2)/16\)，不是三个 shard means 的简单平均。

### 7.2 Mean-only、raw 与续跑

为默认 mean-only 输出建立可用的 base aggregation 路径；不能从 mean 伪造 per-vector 数据。小规模调试保存 raw，用于检查完整层聚合；大规模实验保存足够重建 base sum/count 的数据即可。

metadata 至少包含 configuration、seed、base index、probe 区间、实际 count、scheme、ordering/plan、算法版本、shape、solver/flow 参数。不同 plan 或新旧 ordering 禁止混合。

第一阶段仍以固定 Np 和完整 base completion 续跑；中断在 base 内的行为必须可重复且不重复计权。以后扩展 HP2→32 时，旧 2 个和新 30 个必须合并为一个 32-vector base，不能算成两个独立样本，也不能跳过已标为 HP2 完成的 base。

### 7.3 保持物理定义和实验可比性

source generation 使用全局坐标，保持 even-odd shape 和 backend 广播。相同 config/seed/base 的 source 不应随 batch 或 MPI decomposition 改变。实际归约顺序可能引起舍入差，测试应区分 source 相同与结果浮点容差。

不修改已确认的 EMT kernel、solver 精度、flow steps 和 observable 定义。ringed kinetic 的线性测量与非线性 ringed factor 后处理分开；同一 configuration 的 stochastic SEM 也不能与 ensemble 的 gauge fluctuations 混为一谈。

## 8. 实施顺序与验收：每一步回答一个问题

### 阶段 A：数学构造是否正确

先写 CPU reference tests，再接入 GPU：

1. 复现第 3 节旧 HP 反例，确认测试能识别错误距离排序。
2. 新 HP 仅检查 HP2、HP32、HP512 的距离 1/3/7 保证和完整 Walsh 消去核；验证运行参数拒绝集合之外的数量。
3. Multiplier 检查实际跨界 pairs；9 色失败案例必须被拒绝，固定候选必须通过。
4. 小矩阵验证完整 basis、Fourier-average 与 mask-sum 恒等式、实际 count 归一化。
5. 随机 trace 测试以已知 trace 为基准，用统计误差判断，不要求有限样本精确等于 trace。

验收：保存可重跑的脚本和结果；搜索器报告“找到方案”不能代替独立 checker。

### 阶段 B：工作流是否正确

新增 HP ordering 与 multiplier 分支，补齐 base aggregation、metadata 和固定目标 resume。检查所有 probes 确实运行、计数正确、无旧新混合。

验收：不同 batch 切分、raw/mean-only 聚合、中断续跑与连续运行结果一致；同一 source 跨至少两种可用 MPI decomposition 一致。暂不要求跨 HP 层复用。

### 阶段 C：8^4 实际 EMT 是否正确

先比较相同 base 下 M1/HP2，应在数值精度内一致。再比较普通 noise、M2、HP32，覆盖 kinetic、T44 和实际 flow times；HP512 完整层的小格子集成也须通过后，才能进入大格子性能测试。均值应统计相容，solver 残差应符合原容差。

可从约 32 个独立 bases 的 pilot 起步，但这不是固定充分样本数。保留配对 seeds，按 base 整体重采样估计效率比不确定性；区间过宽则增加样本，不能用一两个 bases 宣布优劣。

### 阶段 D：64^4 是否值得生产使用

对目标 shape 重新验证 plans，再比较 M2/HP32，预算允许时比较 M4/HP512。记录初始化、warm-up、反演、flow、收缩和 I/O，给出各 observable 的 V×C 及不确定性。

若用 pilot 选择 sigma，应在独立保留数据上评价，避免选择偏差。只有确认正确性且效率收益稳定，才考虑调整生产默认方案；不因小格子成功就跳过大格子成本验证。

本轮明确留待后续的内容：10 色方案复现、HP 跨层复用、variance-aware sigma 搜索、任意边长扩展。它们不应阻塞已验证基本算法的有限测试。

## 9. 文献与证据边界

### R1：HP 的基础思想

Stathopoulos, Laeuchli, Orginos (2013), *Hierarchical probing for estimating the trace of the matrix inverse on toroidal lattices*, SIAM J. Sci. Comput. 35, S299–S322。[原文](https://arxiv.org/abs/1302.4018)

支持利用几何邻域、递归着色和 Hadamard ordering 构造可复用 probing 的方法背景。本文关于本地旧实现的反例和新 nested bits 的证明是独立分析，不归称为论文逐行算法。

### R2：Multiplier 与待复现结论

Papace, Frommer, Jimenez-Merchan, Lang, Ramirez-Hidalgo, Schneider (2026), *Probing and graph coloring techniques for trace estimation in Lattice QCD*。[预印本 v1](https://arxiv.org/html/2606.00394v1)

§2 的颜色 mask、§3.2 的 multiplier 与 Table 1 是相关依据。10 色结果的具体 plan 和边界实现仍需核对；不能把其方差优势的论述未经条件检查就套用于本文的完整 HP 层或 flowed EMT。

### R3：任意边长的 HP 扩展

Laeuchli, Stathopoulos (2020), *Extending Hierarchical Probing for Computing the Trace of Matrix Inverses*, SIAM J. Sci. Comput. 42, A1459–A1485。[论文页面](https://doi.org/10.1137/18M1176427)

作为非二次幂格子的后续扩展方向，不作为直接删除初版 shape 限制的理由。

### R4：Observable 会改变需要关注的邻域

Switzer, Stathopoulos, Romero, Orginos (2022), *Probing for the Trace Estimation of a Permuted Matrix Inverse Corresponding to a Lattice Displacement*。[原文](https://arxiv.org/abs/2106.01275)

支持按实际带位移算子设计 probing 的思路，不直接证明某个 multiplier plan 对本项目 flow 后 EMT 最优。

### R5：Disconnected-loop 的其他降噪路线

Bali, Collins, Schäfer (2010), *Effective noise reduction techniques for disconnected loops in Lattice QCD*, Comput. Phys. Commun. 181, 1570–1583。[原文](https://arxiv.org/abs/0910.3970)

用于了解其他降噪方法。低模、dilution 等若后续考虑，应各自做成本测试，而不是与本轮 HP 修复同时更改。

## 10. 最终判断

现有 HP 应修正的是空间层次排序，而不是据此否定全部旧测量；新算法通过 checkerboard 起始的 nested label，给出明确且可检验的距离消去结构。

Multiplier 的数学机制有效，减少 vectors 的潜力值得测试，但必须先解决周期验证、完整 basis、归一化和统计接口。我的建议是：先建立正确 HP 基线，再加入固定 multiplier plans，最终用实际 EMT 的误差与计算成本决定采用哪种方法，而不是仅凭方法名称或颜色数作决定。

## 11. 本轮实施结果与验证边界（2026-09-12）

### 已实现

- EMT 新 HP 默认 ordering 为 `nested_xyzt_checkerboard_v1`，只接受 2、32、512；验证四维二次幂 shape 与对应尺度条件。旧 ordering helper 及独立 ringed/qTMD 默认保留，普通 noise 默认输出身份不变。
- Multiplier 增加独立周期 checker、rank-0 plan 读取/验证与广播、完整 Fourier source 循环、scheme-specific 计数和 metadata。实际 gauge shape 必须与 plan 匹配。
- 固定 JSON plans：8^4 的 M1/M2、64^4 的 M4；周期验证全部通过。首轮按已确认选择不开发 distance-3/7 搜索。
- 后续测试新增 8^4 distance-6：411 色、sigma=(1,8,64,102)。线性模着色类内的最少颜色数由独立有界穷举确定；实际同色点对检查得到最短距离 7。搜索证明与复现命令见 [M6_SEARCH.md](application/EMT_disconnected_1pt/probing_plans/M6_SEARCH.md)，不是全体非线性着色或真实 EMT 效率最优的结论。每个 base 需要 411 次反演，尚未运行 GPU 测试。
- 后续测试新增 8^4 distance-8：829 色、sigma=(1,8,246,310)。由 317 点 clique 下界与 317–828 色穷举排除，确定为线性模着色类内最少颜色；独立同色点对检查的最短距离为 9。每个 base 需要 829 次反演，GPU/MPI 测试尚未运行。见 [M8_SEARCH.md](application/EMT_disconnected_1pt/probing_plans/M8_SEARCH.md)。
- `aggregate_emt_quark_1pt_bases()` 支持 raw/mean-only，按实际 shard count 聚合完整 bases，独立写出 `.bases.h5` 的 base、avg、实部/虚部 SEM 和线性 kinetic。一个 base 的 SEM 为 NaN，不伪造独立 samples；原 raw finalizer 行为保留。
- 缺失、重复或重叠区间、混合 plan/order、无效 count、非有限数据被拒绝；失败不覆盖已有聚合产物。续跑仍是固定目标、完整 base；中断 base 重算。

### 可复现验证

在 develop 根目录、具有 NumPy/h5py/opt_einsum/pytest 的 Python 环境中：

```bash
python -m pytest tests/test_disconnected_probing_math.py -q
python tests/run_probing_cpu_tests.py
```

第一条只运行独立几何和代数测试。第二条明确使用 GPU/MPI 导入 stubs，运行实际 source、EMT 控制、HDF5、resume 和聚合代码的 CPU/mock 回归；gamma 矩阵来自本地 PyQUDA source，不把这类测试称为 GPU 验证。

本轮实测：独立数学测试 **9 passed**；包含这些数学测试的扩展 CPU/mock 回归 **107 passed**。变更 Python 文件的语法编译、GPU 入口 `--help` 与 `git diff --check` 均通过。测试使用当前 Jupyter 的 Python 3.12，pytest 仅安装在 `/tmp/xgao-probing-test-deps`；未改动 Frontier 的 `software/venv`。

覆盖：新 HP 的参考着色与距离 1/3/7、旧排序反例、Walsh/Fourier 正交性、周期 9 色反例、Fourier-average/mask-sum 恒等式、精确有限噪声枚举、完整 source 遍历、batch/partition 一致性、6+6+4 加权、raw/mean 聚合一致性、固定目标中断重跑、单 base SEM 及共享 ringed/qTMD 回归。

### 尚未执行或未包含

当前执行主机是 Jupyter，缺少 Frontier ROCm 模块及对应 PyQUDA 扩展，不能执行真实 HIP/MPI。已提供 `tests/check_disconnected_probing_gpu.py`：在已分配 GPU 中检查 M1/HP2 source 一致性和全局 source hash，可选运行 8^4 完整 base EMT 与一个 flow interval。须用至少两种 MPI geometry 验证，并对 HP2/M1 的实际 EMT 输出作数值检查，之后才能称为 GPU 集成验证通过。

本轮没有提交作业、修改物理 kernel、搜索 distance-3/7、实现 HP 跨层复用或宣称实际性能提升。严格等消去半径的主对照应是 HP2/M1、HP32/M3、HP512/M7；后两组尚缺 plans，不能用当前 M2/M4 结果替代。

完整接口、输出说明和 GPU 命令见 [PROBING.md](application/EMT_disconnected_1pt/PROBING.md)。
