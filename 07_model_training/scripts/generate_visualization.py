#!/usr/bin/env python3
import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# 1. Read test predictions
predictions = []
with open(os.path.join(OUTPUT_DIR, 'test_predictions_weekly.csv'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        predictions.append({
            'date': row['data_date'],
            'y_true': float(row['y_true']),
            'y_pred': float(row['y_pred']),
            'abs_error': float(row['abs_error'])
        })

# 2. Read optuna results
optuna_results = []
best_trial_idx = -1
best_mda = 0
with open(os.path.join(OUTPUT_DIR, 'optuna_hyperparam_results_weekly.csv'), 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        trial_num = int(row['number'])
        mda = float(row['mda'])
        if mda > best_mda:
            best_mda = mda
            best_trial_idx = trial_num
        optuna_results.append({
            'number': trial_num,
            'mda': mda,
            'mape': float(row['mape']),
            'value': float(row['value'])
        })

# 3. Read final evaluation
with open(os.path.join(OUTPUT_DIR, 'final_evaluation_weekly.json'), 'r', encoding='utf-8') as f:
    final_eval = json.load(f)

# 4. Read best params
with open(os.path.join(OUTPUT_DIR, 'best_params_weekly.json'), 'r', encoding='utf-8') as f:
    best_params = json.load(f)

# 5. Read optimization summary
with open(os.path.join(OUTPUT_DIR, 'optimization_summary_weekly.json'), 'r', encoding='utf-8') as f:
    opt_summary = json.load(f)

# Generate HTML using string concatenation to avoid f-string conflicts
html_parts = []

html_parts.append('''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>周度模型训练结果可视化</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
    background: #f5f7fa;
    color: #333;
    padding: 20px;
    line-height: 1.6;
}
.container {
    max-width: 1400px;
    margin: 0 auto;
}
h1 {
    text-align: center;
    color: #1e40af;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 30px;
}
.cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}
.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.card h3 {
    color: #475569;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 15px;
}
.metric {
    margin-bottom: 12px;
}
.metric-label {
    color: #64748b;
    font-size: 13px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #1e40af;
}
.metric-value.green { color: #059669; }
.metric-value.red { color: #dc2626; }
.metric-diff {
    font-size: 13px;
    margin-top: 4px;
}
.params-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    font-size: 13px;
}
.param-item {
    display: flex;
    justify-content: space-between;
}
.param-name { color: #64748b; }
.param-value { font-weight: 600; }
.chart-container {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.chart-title {
    font-size: 16px;
    font-weight: 600;
    color: #1e40af;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid #f1f5f9;
}
.chart {
    width: 100%;
    height: 400px;
}
.grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}
@media (max-width: 900px) {
    .grid-2 { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="container">
    <h1>铁矿石进口库存预测 - 周度模型训练结果</h1>
    <p class="subtitle">ID00186052 · 45港口综合 · 200次Optuna超参优化</p>

    <!-- 概览卡片 -->
    <div class="cards-grid">
        <div class="card">
            <h3>训练集表现</h3>
            <div class="metric">
                <div class="metric-label">方向准确率 MDA</div>
                <div class="metric-value">''')
html_parts.append(f"{final_eval['train_mda']:.2%}")
html_parts.append('''</div>
            </div>
            <div class="metric">
                <div class="metric-label">平均绝对百分比误差 MAPE</div>
                <div class="metric-value">''')
html_parts.append(f"{final_eval['train_mape']:.2%}")
html_parts.append('''</div>
            </div>
            <div class="metric">
                <div class="metric-label">平均绝对误差 MAE（万吨）</div>
                <div class="metric-value">''')
html_parts.append(f"{final_eval['train_mae']:.2f}")
html_parts.append('''</div>
            </div>
            <div class="metric-label">训练样本数: ''')
html_parts.append(str(final_eval['train_rows']))
html_parts.append(''' 条</div>
        </div>

        <div class="card">
            <h3>测试集表现（泛化能力）</h3>
            <div class="metric">
                <div class="metric-label">方向准确率 MDA</div>
                <div class="metric-value green">''')
html_parts.append(f"{final_eval['test_mda']:.2%}")
html_parts.append('''</div>
                <div class="metric-diff red">过拟合下降: ''')
html_parts.append(f"{final_eval['train_mda']-final_eval['test_mda']:.2%}")
html_parts.append('''</div>
            </div>
            <div class="metric">
                <div class="metric-label">平均绝对百分比误差 MAPE</div>
                <div class="metric-value green">''')
html_parts.append(f"{final_eval['test_mape']:.2%}")
html_parts.append('''</div>
                <div class="metric-diff red">增加: ''')
html_parts.append(f"{final_eval['test_mape']-final_eval['train_mape']:.2%}")
html_parts.append('''</div>
            </div>
            <div class="metric">
                <div class="metric-label">平均绝对误差 MAE（万吨）</div>
                <div class="metric-value green">''')
html_parts.append(f"{final_eval['test_mae']:.2f}")
html_parts.append('''</div>
            </div>
            <div class="metric-label">测试样本数: ''')
html_parts.append(str(final_eval['test_rows']))
html_parts.append(''' 条</div>
        </div>

        <div class="card">
            <h3>最优超参数</h3>
            <div class="params-grid">
                <div class="param-item"><span class="param-name">num_leaves</span><span class="param-value">''')
html_parts.append(str(best_params['num_leaves']))
html_parts.append('''</span></div>
                <div class="param-item"><span class="param-name">max_depth</span><span class="param-value">''')
html_parts.append(str(best_params['max_depth']))
html_parts.append('''</span></div>
                <div class="param-item"><span class="param-name">learning_rate</span><span class="param-value">''')
html_parts.append(f"{best_params['learning_rate']:.4f}")
html_parts.append('''</span></div>
                <div class="param-item"><span class="param-name">n_estimators</span><span class="param-value">''')
html_parts.append(str(best_params['n_estimators']))
html_parts.append('''</span></div>
                <div class="param-item"><span class="param-name">min_child_samples</span><span class="param-value">''')
html_parts.append(str(best_params['min_child_samples']))
html_parts.append('''</span></div>
                <div class="param-item"><span class="param-name">subsample</span><span class="param-value">''')
html_parts.append(f"{best_params['subsample']:.4f}")
html_parts.append('''</span></div>
                <div class="param-item"><span class="param-name">colsample_bytree</span><span class="param-value">''')
html_parts.append(f"{best_params['colsample_bytree']:.4f}")
html_parts.append('''</span></div>
                <div class="param-item"><span class="param-name">reg_lambda</span><span class="param-value">''')
html_parts.append(f"{best_params['reg_lambda']:.6f}")
html_parts.append('''</span></div>
            </div>
        </div>
    </div>

    <!-- 预测结果时间序列图 -->
    <div class="chart-container">
        <div class="chart-title">1. 测试集预测结果时间序列（真实值 vs 预测值）</div>
        <div id="predictionChart" class="chart"></div>
    </div>

    <div class="grid-2">
        <!-- Optuna优化过程图 -->
        <div class="chart-container">
            <div class="chart-title">2. Optuna超参优化过程（MDA变化）</div>
            <div id="optunaChart" class="chart"></div>
        </div>

        <!-- 误差分布图 -->
        <div class="chart-container">
            <div class="chart-title">3. 绝对误差分布直方图</div>
            <div id="errorChart" class="chart"></div>
        </div>
    </div>

    <!-- 方向预测结果可视化 -->
    <div class="chart-container">
        <div class="chart-title">4. 涨跌方向预测结果（绿色=正确，红色=错误）</div>
        <div id="directionChart" class="chart"></div>
    </div>
</div>

<script>
const PREDICTIONS = ''')
html_parts.append(json.dumps(predictions))
html_parts.append(''';
const OPTUNA_RESULTS = ''')
html_parts.append(json.dumps(optuna_results))
html_parts.append(''';
const BEST_TRIAL = ''')
html_parts.append(str(best_trial_idx))
html_parts.append(''';
const BEST_MDA = ''')
html_parts.append(str(best_mda))
html_parts.append(''';

// =============== 1. 预测结果时间序列图 ===============
const predictionChart = echarts.init(document.getElementById('predictionChart'));
const predictionDates = PREDICTIONS.map(p => p.date);
const yTrue = PREDICTIONS.map(p => p.y_true);
const yPred = PREDICTIONS.map(p => p.y_pred);
const absErrors = PREDICTIONS.map(p => p.abs_error);

// 计算涨跌方向是否正确
const directionData = [];
for (let i = 1; i < PREDICTIONS.length; i++) {
    const trueDir = PREDICTIONS[i].y_true > PREDICTIONS[i-1].y_true ? 1 : -1;
    const predDir = PREDICTIONS[i].y_pred > PREDICTIONS[i-1].y_pred ? 1 : -1;
    const correct = trueDir === predDir;
    directionData.push({
        date: PREDICTIONS[i].date,
        correct: correct,
        value: correct ? 1 : 0
    });
}

predictionChart.setOption({
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    legend: { data: ['真实库存', '预测库存', '绝对误差'], bottom: 0 },
    grid: { top: 60, right: 80, bottom: 80, left: 80 },
    xAxis: { type: 'category', data: predictionDates, axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: [
        { type: 'value', name: '库存（万吨）', position: 'left' },
        { type: 'value', name: '绝对误差', position: 'right' }
    ],
    series: [
        {
            name: '真实库存',
            type: 'line',
            data: yTrue,
            smooth: true,
            itemStyle: { color: '#3b82f6' },
            lineStyle: { width: 2 }
        },
        {
            name: '预测库存',
            type: 'line',
            data: yPred,
            smooth: true,
            itemStyle: { color: '#f59e0b' },
            lineStyle: { width: 2 }
        },
        {
            name: '绝对误差',
            type: 'bar',
            yAxisIndex: 1,
            data: absErrors,
            itemStyle: { color: 'rgba(239,68,68,0.5)' }
        }
    ]
});

// =============== 2. Optuna优化过程图 ===============
const optunaChart = echarts.init(document.getElementById('optunaChart'));
const trialNumbers = OPTUNA_RESULTS.map(r => r.number);
const mdaValues = OPTUNA_RESULTS.map(r => r.mda);

// 计算最优MDA轨迹
let bestSoFar = 0;
const bestMdaTrace = [];
for (let i = 0; i < mdaValues.length; i++) {
    bestSoFar = Math.max(bestSoFar, mdaValues[i]);
    bestMdaTrace.push(bestSoFar);
}

optunaChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['单次试验MDA', '最优MDA轨迹'], bottom: 0 },
    grid: { top: 60, right: 60, bottom: 80, left: 80 },
    xAxis: { type: 'category', name: 'Trial编号', data: trialNumbers },
    yAxis: { type: 'value', name: 'MDA', min: 0.5, max: 1 },
    series: [
        {
            name: '单次试验MDA',
            type: 'scatter',
            data: mdaValues,
            itemStyle: { color: 'rgba(59,130,246,0.6)' }
        },
        {
            name: '最优MDA轨迹',
            type: 'line',
            data: bestMdaTrace,
            smooth: true,
            itemStyle: { color: '#059669' },
            lineStyle: { width: 3 }
        },
        {
            name: '最佳Trial',
            type: 'scatter',
            data: (() => {
                const arr = new Array(mdaValues.length).fill(null);
                arr[BEST_TRIAL] = BEST_MDA;
                return arr;
            })(),
            itemStyle: { color: '#dc2626', symbolSize: 12 }
        }
    ],
    markLine: {}
});

// =============== 3. 误差分布图 ===============
const errorChart = echarts.init(document.getElementById('errorChart'));
const errors = PREDICTIONS.map(p => p.abs_error).sort((a,b) => a-b);
const p50 = errors[Math.floor(errors.length * 0.5)];
const p75 = errors[Math.floor(errors.length * 0.75)];
const p95 = errors[Math.floor(errors.length * 0.95)];

errorChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { top: 60, right: 60, bottom: 80, left: 60 },
    xAxis: { name: '绝对误差（万吨）' },
    yAxis: { name: '频数' },
    series: [{
        type: 'histogram',
        data: errors,
        itemStyle: { color: '#3b82f6', opacity: 0.7 }
    }]
});

// =============== 4. 方向预测结果可视化 ===============
const directionChart = echarts.init(document.getElementById('directionChart'));
const dirDates = directionData.map(d => d.date);

directionChart.setOption({
    tooltip: {
        formatter: function(params) {
            var result = params.name;
            result += '<br>' + params.data.date;
            result += '<br>预测方向: ' + (params.data.correct ? '正确 ✓' : '错误 ✗');
            return result;
        }
    },
    grid: { top: 60, right: 60, bottom: 80, left: 60 },
    xAxis: { type: 'category', data: dirDates, axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: 'value', min: 0, max: 1, show: false },
    series: [{
        type: 'bar',
        data: directionData.map(d => ({
            value: 1,
            date: d.date,
            correct: d.correct,
            itemStyle: { color: d.correct ? '#059669' : '#dc2626' }
        })),
        barWidth: '80%'
    }]
});

// 响应式
window.addEventListener('resize', function() {
    predictionChart.resize();
    optunaChart.resize();
    errorChart.resize();
    directionChart.resize();
});
</script>
</body>
</html>''')

# Write the HTML file
output_path = os.path.join(OUTPUT_DIR, 'weekly_model_results.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(''.join(html_parts))

print(f"HTML visualization generated at: {output_path}")
print("Open this file in your browser to view the results!")
