#!/usr/bin/env python3
"""精简 NPUKernelBench 测试用例：每个算子保留 <=10 条。

精简原则：
1. 按整行 JSON 精确去重（保留首次出现）；
2. 去重后 <= 10 条的算子按实际数量全量保留；
3. 超过 10 条的：先锁定总规模最小 / 中位数 / 最大 3 条边界用例，
   其余名额用贪心最大覆盖选择，特征优先级：
   数值类型组合(100) > 属性取值组合(10) > 张量维度模式(5) > 规模档位(2)，
   同分时偏好接近中位数规模的主流用例；
4. 支持 tensor（shape）与 tensor_list（shapes）两种输入形式；
5. 结果按总规模从小到大排序，以原 JSONL 格式写入 <原名>_simple.json。
"""
import json
import glob
import math
import sys

K = 10

_warned_types = set()


def numel(shape):
    if not shape:
        return 1
    n = 1
    for d in shape:
        n *= d
    return n


def tensors_of(c):
    """统一提取 tensor / tensor_list 输入为 (dtype, shapes) 列表。"""
    out = []
    for i in c['inputs']:
        t = i.get('type')
        if t == 'tensor':
            out.append((i['dtype'], [i.get('shape') or []]))
        elif t == 'tensor_list':
            out.append((i['dtype'], [s or [] for s in i.get('shapes', [])]))
        elif t != 'attr' and t not in _warned_types:
            _warned_types.add(t)
            print(f"WARNING: 未知输入类型 {t!r}（字段 {i.get('name')!r}），"
                  f"该条目未参与特征提取，请检查是否需要扩展 tensors_of()",
                  file=sys.stderr)
    return out


def case_features(c):
    ts = tensors_of(c)
    at = [i for i in c['inputs'] if i.get('type') == 'attr']
    dt = tuple(d for d, _ in ts)
    av = tuple((a['name'], json.dumps(a.get('value'), sort_keys=True)) for a in at)
    total = sum(numel(s) for _, shapes in ts for s in shapes)
    ranks = tuple(len(s) for _, shapes in ts for s in shapes)
    return {
        'dt': dt,
        'av': av,
        'total': total,
        'bucket': int(math.log2(max(total, 1))),
        'ranks': ranks,
    }


def select(cases, k=K):
    # 1. 精确去重，保持原顺序
    seen, uniq = set(), []
    for c in cases:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    if len(uniq) <= k:
        return sorted(uniq, key=lambda c: case_features(c)['total'])

    feats = [case_features(c) for c in uniq]
    totals = [f['total'] for f in feats]

    # 2. 锁定最小 / 中位数 / 最大规模用例
    order = sorted(range(len(uniq)), key=lambda i: totals[i])
    median_target = sorted(totals)[len(totals) // 2]
    seeds = {
        order[0],  # 最小
        min(range(len(uniq)), key=lambda i: (abs(totals[i] - median_target), i)),  # 中位数
        order[-1],  # 最大
    }
    chosen = list(dict.fromkeys(i for i in (order[0],
                                            min(range(len(uniq)), key=lambda i: (abs(totals[i] - median_target), i)),
                                            order[-1])))
    covered = {key: {feats[i][key] for i in chosen} for key in ('dt', 'av', 'ranks', 'bucket')}
    chosen_set = set(chosen)

    log_sizes = sorted(math.log2(max(t, 1)) for t in totals)
    median_log = log_sizes[len(log_sizes) // 2]

    def score(i):
        f = feats[i]
        s = 0.0
        if f['dt'] not in covered['dt']:
            s += 100  # 数值类型覆盖优先
        if f['av'] not in covered['av']:
            s += 10   # 属性取值（mode / dim / threshold 等语义分支）
        if f['ranks'] not in covered['ranks']:
            s += 5    # 张量维度模式
        if f['bucket'] not in covered['bucket']:
            s += 2    # 规模档位
        s -= abs(math.log2(max(f['total'], 1)) - median_log) * 0.1  # 偏好主流规模
        return s

    # 3. 贪心补足剩余名额
    while len(chosen) < k:
        best, best_s = None, None
        for i in range(len(uniq)):
            if i in chosen_set:
                continue
            s = score(i)
            if best_s is None or s > best_s:
                best_s, best = s, i
        chosen.append(best)
        chosen_set.add(best)
        for key in covered:
            covered[key].add(feats[best][key])

    selected = [uniq[i] for i in chosen]
    selected.sort(key=lambda c: case_features(c)['total'])
    return selected


def main():
    files = sorted(f for f in glob.glob('*.json') if '_simple' not in f)
    print(f"{'file':58s} {'orig':>5s} {'sel':>4s}  dtypes kept")
    for path in files:
        cases = [json.loads(line) for line in open(path) if line.strip()]
        picked = select(cases)
        out = path.replace('.json', '_simple.json')
        with open(out, 'w') as fo:
            for c in picked:
                fo.write(json.dumps(c) + '\n')
        dts = sorted({case_features(c)['dt'] for c in picked})
        dt_str = '; '.join('/'.join(d) for d in dts)[:70]
        print(f"{path:58s} {len(cases):5d} {len(picked):4d}  {dt_str}")


if __name__ == '__main__':
    main()
