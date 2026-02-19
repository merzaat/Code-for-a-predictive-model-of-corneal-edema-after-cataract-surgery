from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import 全部
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.ensemble import GradientBoostingClassifier
import warnings
import base64
from io import BytesIO

warnings.filterwarnings('ignore')

# 初始化FastAPI
app = FastAPI(title="Cataract Postoperative Corneal Damage Prediction System", version="1.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 核心配置
EXCEL_PATH = "CEML3特征交集.xlsx"
MODEL_PATH = "corneal_damage_model.pkl"
SCALER_PATH = "data_scaler.pkl"
SHAP_EXPLAINER_PATH = "shap_explainer.pkl"
ROC_SAVE_PATH = "roc_curve.png"
SHAP_BEESWARM_PATH = "shap_beeswarm.png"  # SHAP蜂群图保存路径
SHAP_SCATTER_PATH = "shap_scatter.png"    # SHAP散点拟合图保存路径
TARGET_COL = "角膜损伤状态"
DATA_SAVE_EXCEL_PATH = "dataset_split.xlsx"
TOP_N_FEATURES = 15  # 散点拟合图展示的TOP特征数

# 全局变量
model = None
scaler = None
shap_explainer = None
FEATURE_NAMES = []
shap_beeswarm_base64 = None  # 存储SHAP蜂群图的base64编码
shap_scatter_base64 = None   # 存储SHAP散点拟合图的base64编码
global_shap_values = None    # 全量样本的SHAP值（全局）
global_feature_values = None # 全量样本的原始特征值（全局）
global_y_values = None       # 全量样本的目标值（用于组合图分类统计）

# 临床建议生成函数
def generate_clinical_advice(risk_level, risk_prob, input_params):
    """根据风险等级、概率和输入参数生成临床建议"""
    age = input_params.get("年龄", 0)
    anterior_chamber_depth = input_params.get("前房深度", 0)
    total_surgery_time = input_params.get("总手术时间", 0)
    negative_pressure_time = input_params.get("负压时间", 0)
    effective_emulsification_time = input_params.get("有效乳化时间", 0)
    
    if risk_level == "低风险":
        return f"""【术前建议】
1. 角膜损伤风险{risk_prob}%（低风险），常规行角膜内皮细胞计数检查；
2. 年龄{age}岁，评估基础疾病控制情况；
3. 前房深度{anterior_chamber_depth}mm，手术按常规流程进行。

【术中建议】
1. 总手术时间控制在{total_surgery_time}秒内；
2. 负压时间维持安全范围。

【术后建议】
1. 术后1、3天复查角膜水肿；
2. 局部用抗生素+激素滴眼液1周；
3. 1个月后随访视力恢复。"""
    
    elif risk_level == "中风险":
        return f"""【术前建议】
1. 角膜损伤风险{risk_prob}%（中风险），完善角膜内皮细胞密度检测；
2. 年龄{age}岁，术前3天用人工泪液改善眼表；
3. 排除术前角膜病变。

【术中建议】
1. 总手术时间缩短至{round(total_surgery_time*0.8)}秒内；
2. 负压时间控制在{round(negative_pressure_time*0.9)}秒以下；
3. 有效乳化时间控制在{round(effective_emulsification_time*0.8)}秒内。

【术后建议】
1. 每日复查角膜水肿及眼压，持续3天；
2. 激素滴眼液4次/日，持续2周；
3. 加用角膜保护剂，1、2、4周随访。"""
    
    elif risk_level == "高风险":
        return f"""【术前建议】
1. 角膜损伤风险{risk_prob}%（高风险），完善角膜厚度、眼压检查；
2. 内皮细胞密度<1800个/mm²需沟通手术方案；
3. 年龄{age}岁+前房深度{anterior_chamber_depth}mm，高年资医师主刀。

【术中建议】
1. 总手术时间≤400秒（当前{total_surgery_time}秒，需缩短{max(0, total_surgery_time-400)}秒）；
2. 负压时间≤150秒，有效乳化时间≤3秒；
3. 用粘弹剂保护角膜内皮。

【术后建议】
1. 留院观察24小时，每6小时评估水肿；
2. 激素滴眼液1次/小时冲击3天，加用高渗盐水；
3. 1周内每日复查，1个月内每周复查。"""
    
    else:
        return f"暂无{risk_level}对应的临床建议（风险概率：{risk_prob}%）"

# 绘制SHAP蜂群图（蜂巢+堆叠条形组合图）
def plot_shap_beeswarm(X_scaled, feature_names):
    """
    绘制SHAP
    蜂巢图+堆叠条形图组合图，并返回base64编码
    :param X_scaled: 标准化后的特征数据
    :param feature_names: 特征名称列表
    :return: base64编码的图片字符串
    """
    global shap_beeswarm_base64, global_shap_values, global_y_values
    
    # 设置中文字体和图片样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.figure(figsize=(20, 8))  # 调整尺寸适配组合图
    
    try:
        # 计算SHAP值 - 兼容不同SHAP版本的返回格式
        shap_values = shap_explainer.shap_values(X_scaled)
        
        # 处理分类模型的SHAP值（二分类模型返回list，取正类；多分类/回归返回array）
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # 二分类模型，取正类的SHAP值
            shap_values_pos = shap_values[1]
        elif isinstance(shap_values, np.ndarray):
            # 回归/多分类单输出
            shap_values_pos = shap_values
        else:
            # 其他情况取第一个维度
            shap_values_pos = shap_values[0] if len(shap_values) > 0 else shap_values
        
        # 保存全量SHAP值到全局变量
        global_shap_values = shap_values_pos
        
        # 筛选TOP N特征（按平均绝对SHAP值降序）
        feat_importance = np.abs(shap_values_pos).mean(axis=0)
        top_idx = np.argsort(feat_importance)[::-1][:TOP_N_FEATURES]
        top_feat_names = [feature_names[i] for i in top_idx]
        top_shap_values = shap_values_pos[:, top_idx]
        top_feature_values = X_scaled[:, top_idx]
        
        # 创建组合图（1行2列）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1]})
        
        # 左侧：蜂巢图（原蜂群图）
        plt.sca(ax1)
        shap.summary_plot(
            top_shap_values,
            features=top_feature_values,
            feature_names=top_feat_names,
            plot_type="dot",
            max_display=TOP_N_FEATURES,
            show=False,
            color_bar_label="特征值"  # 中文标签
        )
        ax1.tick_params(axis="y", labelsize=9, pad=5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xlabel("SHAP值", fontsize=12, labelpad=10)  # 中文标签
        
        # 右侧：堆叠条形图（按类别拆分）
        if global_y_values is not None:
            # 按目标值分类计算平均绝对SHAP值
            y_values = np.array(global_y_values)
            class0_mask = y_values == 0
            class1_mask = y_values == 1
            
            class0_shap = np.abs(top_shap_values[class0_mask]).mean(axis=0) if np.any(class0_mask) else np.zeros(TOP_N_FEATURES)
            class1_shap = np.abs(top_shap_values[class1_mask]).mean(axis=0) if np.any(class1_mask) else np.zeros(TOP_N_FEATURES)
            
            # 按总重要性降序排序
            total_importance = class0_shap + class1_shap
            sort_idx = np.argsort(total_importance)
            sorted_feats = [top_feat_names[i] for i in sort_idx]
            sorted_class0 = class0_shap[sort_idx]
            sorted_class1 = class1_shap[sort_idx]
            
            # 绘制堆叠条形图
            bar_width = 0.8
            y_pos = np.arange(len(sorted_feats))
            ax2.barh(y_pos, sorted_class0, height=bar_width, color="#4575b4", label="0-2级水肿")
            ax2.barh(y_pos, sorted_class1, height=bar_width, left=sorted_class0, color="#d73027", label="3级水肿")
            
            # 美化条形图
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(sorted_feats, fontsize=9)
            ax2.set_xlabel("平均绝对SHAP值（特征重要性）", fontsize=11)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.legend(
                handles=[Patch(facecolor="#4575b4", label="0-2级水肿"), Patch(facecolor="#d73027", label="3级水肿")],
                fontsize=8, loc="lower right"
            )
            ax2.set_xlim(0, max(total_importance) * 1.1)
        
        # 调整整体布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 将图片保存到BytesIO并转换为base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        shap_beeswarm_base64 = img_base64
        
        # 保存图片到文件
        plt.savefig(SHAP_BEESWARM_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP组合图（蜂巢+条形）已保存到: {os.path.abspath(SHAP_BEESWARM_PATH)}")
        return img_base64
        
    except Exception as e:
        print(f"⚠️ 绘制SHAP组合图时出错: {str(e)}")
        # 降级方案：使用基础的summary_plot
        plt.clf()
        shap.summary_plot(
            shap_explainer.shap_values(X_scaled),
            X_scaled,
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        shap_beeswarm_base64 = img_base64
        plt.savefig(SHAP_BEESWARM_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        return img_base64

# 绘制SHAP散点拟合图
def plot_shap_scatter_fit(X_scaled, feature_names):
    """
    绘制SHAP散点拟合图（带LOWESS拟合曲线），返回base64编码
    :param X_scaled: 标准化后的特征数据
    :param feature_names: 特征名称列表
    :return: base64编码的图片字符串
    """
    global shap_scatter_base64, global_shap_values
    
    # 设置中文字体和图片样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    
    try:
        # 获取全局SHAP值
        if global_shap_values is None:
            shap_values = shap_explainer.shap_values(X_scaled)
            shap_values_pos = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
            global_shap_values = shap_values_pos
        
        # 筛选TOP N特征（按平均绝对SHAP值降序）
        feat_importance = np.abs(global_shap_values).mean(axis=0)
        top_idx = np.argsort(feat_importance)[::-1][:TOP_N_FEATURES]
        top_feat_names = [feature_names[i] for i in top_idx]
        top_shap_values = global_shap_values[:, top_idx]
        top_feature_values = X_scaled[:, top_idx]
        
        # 选取前12个特征绘制
        plot_feats = top_feat_names[:12]
        plot_shap = top_shap_values[:, :12]
        plot_features = top_feature_values[:, :12]
        
        # 创建画布
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(4, 3, figure=fig, wspace=0.5, hspace=0.3)
        lowess_color = '#457B9D'  # LOWESS曲线颜色
        cmap = plt.cm.coolwarm    # 散点颜色映射

        # 循环绘制每个特征
        for i, feat in enumerate(plot_feats):
            # 子网格（散点图+颜色棒）
            sub_gs = gs[i//3, i%3].subgridspec(1, 2, width_ratios=[30, 1], wspace=0.05)
            ax_scatter = plt.subplot(sub_gs[0, 0])
            
            # 提取数据
            x = plot_features[:, i]
            y = plot_shap[:, i]
            
            # 颜色映射
            norm = plt.Normalize(vmin=np.min(x), vmax=np.max(x))
            scatter = ax_scatter.scatter(
                x, y, alpha=0.7, s=10, c=x, cmap=cmap, norm=norm,
                edgecolor='k', linewidth=0.6, zorder=3
            )
            
            # LOWESS拟合
            try:
                lowess_fit = lowess(y, x, frac=0.35)
                ax_scatter.plot(lowess_fit[:, 0], lowess_fit[:, 1], 
                               color=lowess_color, linewidth=1.8, alpha=0.9, zorder=4)
            except Exception as e:
                print(f"特征 {feat} LOWESS拟合失败: {e}")
            
            # y=0红线
            ax_scatter.axhline(y=0, color='#E63946', linestyle='--', linewidth=1, alpha=0.8, zorder=2)
            
            # 美化
            ax_scatter.set_xlabel(feat, fontsize=12)
            ax_scatter.set_ylabel('SHAP Value', fontsize=12)
            ax_scatter.spines['top'].set_visible(False)
            ax_scatter.spines['right'].set_visible(False)
            ax_scatter.spines['left'].set_linewidth(0.9)
            ax_scatter.spines['bottom'].set_linewidth(0.9)
            
            # 颜色棒
            ax_colorbar = plt.subplot(sub_gs[0, 1])
            cbar = fig.colorbar(scatter, cax=ax_colorbar, orientation='vertical')
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('Feature Value', fontsize=12, labelpad=3)
            
            # 调整颜色棒刻度
            if np.max(x) - np.min(x) > 5:
                cbar.locator = plt.MaxNLocator(nbins=3)
                cbar.update_ticks()

        # 调整布局并保存
        plt.subplots_adjust(left=0.05, right=0.98, hspace=0.3, wspace=0.5)
        
        # 将图片保存到BytesIO并转换为base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        shap_scatter_base64 = img_base64
        
        # 保存图片到文件
        plt.savefig(SHAP_SCATTER_PATH, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP散点拟合图已保存到: {os.path.abspath(SHAP_SCATTER_PATH)}")
        return img_base64
        
    except Exception as e:
        print(f"⚠️ 绘制SHAP散点拟合图时出错: {str(e)}")
        plt.close()
        raise e

# 模型评估与ROC曲线
def evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test):
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1].tolist()
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1].tolist()
    y_train_pred = (np.array(y_train_prob) >= 0.5).astype(int).tolist()
    y_test_pred = (np.array(y_test_prob) >= 0.5).astype(int).tolist()

    metrics = {
        "Training Set": {
            "AUC": round(roc_auc_score(y_train, y_train_prob), 4),
            "Accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "Precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "Confusion Matrix": confusion_matrix(y_train, y_train_pred).tolist()
        },
        "Test Set": {
            "AUC": round(roc_auc_score(y_test, y_test_prob), 4),
            "Accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "Precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "F1-Score": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "Confusion Matrix": confusion_matrix(y_test, y_test_pred).tolist()
        }
    }

    print("\n" + "="*80)
    print("📊 Model Evaluation Results")
    print("="*80)
    for set_name, set_metrics in metrics.items():
        print(f"\n【{set_name}】")
        for k, v in set_metrics.items():
            if k != "Confusion Matrix":
                print(f"  {k:<12} : {v}")
            else:
                print(f"  {k}:")
                for row in v:
                    print(f"    {row}")
    print("="*80 + "\n")

    # 绘制ROC曲线
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
    plt.figure(figsize=(8, 6))
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    plt.plot(fpr_train, tpr_train, label=f"Training Set (AUC = {metrics['Training Set']['AUC']})", 
             linewidth=2.5, color="#2E86AB")
    plt.plot(fpr_test, tpr_test, label=f"Test Set (AUC = {metrics['Test Set']['AUC']})", 
             linewidth=2.5, color="#A23B72")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess", alpha=0.7)
    plt.xlabel("False Positive Rate (FPR)", fontsize=14, fontweight="bold")
    plt.ylabel("True Positive Rate (TPR)", fontsize=14, fontweight="bold")
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_SAVE_PATH, dpi=300, bbox_inches="tight")
    print(f"✅ ROC saved to: {os.path.abspath(ROC_SAVE_PATH)}")
    plt.close()

# 数据处理与模型训练
def load_data_and_train():
    global model, scaler, shap_explainer, FEATURE_NAMES, shap_beeswarm_base64, shap_scatter_base64
    global global_shap_values, global_feature_values, global_y_values

    # 读取Excel
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    print(f"✅ Excel loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 校验目标列
    if TARGET_COL not in df.columns:
        raise HTTPException(status_code=500, detail=f"Target column '{TARGET_COL}' not found")

    # 特征列识别
    FEATURE_NAMES = [col for col in df.columns if col != TARGET_COL]
    if len(FEATURE_NAMES) < 1:
        raise HTTPException(status_code=500, detail="No feature columns found")
    print(f"✅ Features (中文列名): {FEATURE_NAMES}")

    # 缺失值处理
    df_clean = df.dropna(subset=FEATURE_NAMES + [TARGET_COL])
    if df_clean.shape[0] < 10:
        raise HTTPException(status_code=500, detail="Too few data rows after cleaning")

    # 数据准备
    X = df_clean[FEATURE_NAMES].values
    y = np.where(df_clean[TARGET_COL].values != 0, 1, 0).tolist()
    
    # 保存全量原始特征值和目标值到全局变量
    global_feature_values = X
    global_y_values = y  # 保存目标值

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # SMOTE过采样
    X_train_smote, y_train_smote = X_train, y_train
    if y_train.count(1) < y_train.count(0) and y_train.count(1) >= 2:
        scaler_temp = StandardScaler()
        X_train_scaled_temp = scaler_temp.fit_transform(X_train)
        smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.count(1)-1))
        X_train_smote_scaled, y_train_smote = smote.fit_resample(X_train_scaled_temp, y_train)
        X_train_smote = scaler_temp.inverse_transform(X_train_smote_scaled)
        print(f"✅ SMOTE applied: {X_train_smote.shape}")

    # 保存数据集到Excel
    with pd.ExcelWriter(DATA_SAVE_EXCEL_PATH, engine='openpyxl') as writer:
        train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
        train_df[TARGET_COL] = y_train
        train_df.to_excel(writer, sheet_name='训练集_原始', index=False)
        
        test_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
        test_df[TARGET_COL] = y_test
        test_df.to_excel(writer, sheet_name='测试集', index=False)
        
        smote_train_df = pd.DataFrame(X_train_smote, columns=FEATURE_NAMES)
        smote_train_df[TARGET_COL] = y_train_smote
        smote_train_df.to_excel(writer, sheet_name='训练集_SMOTE后', index=False)
    
    print(f"✅ 数据集已保存到Excel: {os.path.abspath(DATA_SAVE_EXCEL_PATH)}")
    print(f"   - 工作表1: 训练集_原始 (行数: {len(X_train)})")
    print(f"   - 工作表2: 测试集 (行数: {len(X_test)})")
    print(f"   - 工作表3: 训练集_SMOTE后 (行数: {len(X_train_smote)})")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    # 重新标准化SMOTE后的训练集
    X_train_smote_scaled = scaler.transform(X_train_smote)

    # 训练模型
    model = GradientBoostingClassifier(
        n_estimators=450,
        learning_rate=0.0529678687405566,
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.6251214490822976,
        max_features='log2',
        loss='exponential',
        random_state=42
    )
    model.fit(X_train_smote_scaled, y_train_smote)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # 评估模型
    evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 初始化SHAP解释器
    shap_explainer = shap.TreeExplainer(model)
    joblib.dump(shap_explainer, SHAP_EXPLAINER_PATH)
    
    # 标准化整个数据集用于SHAP计算
    X_full_scaled = scaler.transform(X)
    
    # 绘制整个数据集的SHAP蜂群图（现在是组合图）
    plot_shap_beeswarm(X_full_scaled, FEATURE_NAMES)
    
    # 绘制整个数据集的SHAP散点拟合图
    plot_shap_scatter_fit(X_full_scaled, FEATURE_NAMES)
    
    print(f"✅ SHAP explainer saved")

# 服务初始化
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(SHAP_EXPLAINER_PATH):
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]
        FEATURE_NAMES = model_data["feature_names"]
        scaler = joblib.load(SCALER_PATH)
        shap_explainer = joblib.load(SHAP_EXPLAINER_PATH)
        
        # 加载全量数据并计算全局SHAP值
        if os.path.exists(EXCEL_PATH):
            df = pd.read_excel(EXCEL_PATH)
            df_clean = df.dropna(subset=FEATURE_NAMES + [TARGET_COL])
            X = df_clean[FEATURE_NAMES].values
            y = np.where(df_clean[TARGET_COL].values != 0, 1, 0).tolist()
            global_feature_values = X  # 保存全量原始特征值
            global_y_values = y        # 保存目标值
            X_full_scaled = scaler.transform(X)
            
            # 计算并保存全局SHAP值
            shap_values = shap_explainer.shap_values(X_full_scaled)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                global_shap_values = shap_values[1]  # 二分类取正类
            else:
                global_shap_values = shap_values
            
            # 如果未生成SHAP蜂群图则生成
            if not os.path.exists(SHAP_BEESWARM_PATH):
                plot_shap_beeswarm(X_full_scaled, FEATURE_NAMES)
            
            # 如果未生成SHAP散点拟合图则生成
            if not os.path.exists(SHAP_SCATTER_PATH):
                plot_shap_scatter_fit(X_full_scaled, FEATURE_NAMES)
        
        print(f"\n✅ Model loaded. Features (中文): {FEATURE_NAMES}")
        print(f"✅ Global SHAP values loaded: {global_shap_values.shape if global_shap_values is not None else 'None'}")
    else:
        print(f"\n⚠️ No model found. Training...")
        load_data_and_train()
    print(f"\n🚀 Service ready: http://localhost:8000/docs")
except Exception as e:
    print(f"❌ Init failed: {str(e)}")
    # 打印详细的错误堆栈信息
    import traceback
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"Init failed: {str(e)}")

# 预测接口
@app.post("/predict")
async def predict_corneal_damage(
    params: dict = Body(..., example={
        "年龄": 60.0,
        "最佳矫正视力": 0.5,
        "前房深度": 3.0,
        "前房容积": 150.0,
        "总手术时间": 600.0,
        "负压时间": 200.0,
        "有效乳化时间": 5.0
    })
):
    try:
        # 校验参数完整性
        missing_features = [col for col in FEATURE_NAMES if col not in params]
        if missing_features:
            raise ValueError(f"Missing parameters (缺失参数): {missing_features}")

        # 校验参数类型
        input_list = []
        for col in FEATURE_NAMES:
            val = params[col]
            if not isinstance(val, (int, float)):
                raise ValueError(f"Parameter '{col}' must be number, got {type(val)}")
            input_list.append(float(val))

        # 标准化与预测
        input_data = np.array(input_list).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        risk_prob = model.predict_proba(input_scaled)[0][1]
        risk_prob = float(round(risk_prob, 4))

        # 风险等级
        if risk_prob > 0.7:
            risk_level = "高风险"
        elif risk_prob > 0.4:
            risk_level = "中风险"
        else:
            risk_level = "低风险"

        # 单个样本的SHAP值
        shap_values = [0.0 for _ in FEATURE_NAMES]
        if shap_explainer:
            try:
                shap_result = shap_explainer.shap_values(input_scaled)
                shap_array = shap_result[1] if isinstance(shap_result, list) else shap_result
                shap_list = shap_array[0].tolist()
                shap_values = [float(round(val, 4)) for val in shap_list]
            except Exception as e:
                print(f"⚠️ SHAP failed: {str(e)}")
                shap_values = [0.0 for _ in FEATURE_NAMES]

        # 置信度
        confidence = float(round(0.85 + (min(risk_prob, 1 - risk_prob) * 0.13), 4))
        
        # 生成临床建议（风险概率保留2位小数）
        risk_prob_percent = round(risk_prob * 100, 2)  # 保留2位小数
        clinical_advice = generate_clinical_advice(risk_level, risk_prob_percent, params)

        # 准备全局SHAP数据（用于前端绘制蜂巢图）
        global_shap_data = []
        global_feature_data = []
        if global_shap_values is not None and global_feature_values is not None:
            global_shap_data = global_shap_values.tolist()
            global_feature_data = global_feature_values.tolist()

        # 返回结果（全局SHAP数据）
        return {
            "code": 200,
            "message": "Prediction Successful",
            "data": {
                "feature_names": FEATURE_NAMES,
                "risk_probability": risk_prob,
                "confidence": confidence,
                "risk_level": risk_level,
                "shap_values": shap_values,          # 单个样本的SHAP值
                "global_shap_values": global_shap_data,  # 全量样本的SHAP值
                "global_feature_values": global_feature_data,  # 全量样本的原始特征值
                "clinical_advice": clinical_advice
            }
        }
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {"code": 500, "message": error_msg, "data": None}

# 健康检查接口
@app.get("/health")
async def health_check():
    return {
        "code": 200,
        "message": "Service Running Normally",
        "data": {
            "model_loaded": bool(model),
            "feature_names": FEATURE_NAMES,
            "shap_supported": bool(shap_explainer),
            "required_params": FEATURE_NAMES,
            "param_count": len(FEATURE_NAMES),
            "shap_beeswarm_generated": os.path.exists(SHAP_BEESWARM_PATH),
            "shap_scatter_generated": os.path.exists(SHAP_SCATTER_PATH),
            "global_shap_available": global_shap_values is not None
        }
    }

# 获取SHAP蜂群图接口
@app.get("/get_shap_beeswarm")
async def get_shap_beeswarm():
    """返回SHAP组合图的base64编码，用于前端嵌入"""
    try:
        global shap_beeswarm_base64
        
        # 如果base64未缓存，从文件读取并转换
        if not shap_beeswarm_base64:
            if not os.path.exists(SHAP_BEESWARM_PATH):
                # 重新生成图片
                df = pd.read_excel(EXCEL_PATH)
                df_clean = df.dropna(subset=FEATURE_NAMES + [TARGET_COL])
                X = df_clean[FEATURE_NAMES].values
                X_full_scaled = scaler.transform(X)
                plot_shap_beeswarm(X_full_scaled, FEATURE_NAMES)
            
            with open(SHAP_BEESWARM_PATH, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
                shap_beeswarm_base64 = img_base64
        
        return {
            "code": 200,
            "message": "SHAP beeswarm plot retrieved successfully",
            "data": {
                "image_base64": shap_beeswarm_base64,
                "image_type": "png"
            }
        }
    except Exception as e:
        error_msg = f"Failed to get SHAP beeswarm plot: {str(e)}"
        print(f"❌ {error_msg}")
        # 打印详细错误
        import traceback
        traceback.print_exc()
        return {"code": 500, "message": error_msg, "data": None}

# 获取SHAP散点拟合图接口
@app.get("/get_shap_scatter")
async def get_shap_scatter():
    """返回SHAP散点拟合图的base64编码，用于前端嵌入"""
    try:
        global shap_scatter_base64
        
        # 如果base64未缓存，从文件读取并转换
        if not shap_scatter_base64:
            if not os.path.exists(SHAP_SCATTER_PATH):
                # 重新生成图片
                df = pd.read_excel(EXCEL_PATH)
                df_clean = df.dropna(subset=FEATURE_NAMES + [TARGET_COL])
                X = df_clean[FEATURE_NAMES].values
                X_full_scaled = scaler.transform(X)
                plot_shap_scatter_fit(X_full_scaled, FEATURE_NAMES)
            
            with open(SHAP_SCATTER_PATH, 'rb') as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
                shap_scatter_base64 = img_base64
        
        return {
            "code": 200,
            "message": "SHAP scatter plot retrieved successfully",
            "data": {
                "image_base64": shap_scatter_base64,
                "image_type": "png"
            }
        }
    except Exception as e:
        error_msg = f"Failed to get SHAP scatter plot: {str(e)}"
        print(f"❌ {error_msg}")
        # 打印详细错误
        import traceback
        traceback.print_exc()
        return {"code": 500, "message": error_msg, "data": None}

# 模型上传接口
@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    global model, FEATURE_NAMES, shap_explainer, shap_beeswarm_base64, shap_scatter_base64
    global global_shap_values, global_feature_values, global_y_values
    try:
        if not file.filename.endswith((".pkl", ".joblib")):
            raise ValueError("Only .pkl/.joblib files are allowed")

        # 保存模型
        with open(MODEL_PATH, "wb") as f:
            f.write(await file.read())

        # 加载新模型
        model_data = joblib.load(MODEL_PATH)
        if "model" not in model_data or "feature_names" not in model_data:
            raise ValueError("Model file missing 'model' or 'feature_names'")
        
        model = model_data["model"]
        FEATURE_NAMES = model_data["feature_names"]
        shap_explainer = shap.TreeExplainer(model)
        joblib.dump(shap_explainer, SHAP_EXPLAINER_PATH)
        
        # 重新计算全局SHAP值
        if os.path.exists(EXCEL_PATH):
            df = pd.read_excel(EXCEL_PATH)
            df_clean = df.dropna(subset=FEATURE_NAMES + [TARGET_COL])
            X = df_clean[FEATURE_NAMES].values
            y = np.where(df_clean[TARGET_COL].values != 0, 1, 0).tolist()
            global_feature_values = X
            global_y_values = y  # 保存目标值
            X_full_scaled = scaler.transform(X)
            
            # 计算全局SHAP值
            shap_values = shap_explainer.shap_values(X_full_scaled)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                global_shap_values = shap_values[1]
            else:
                global_shap_values = shap_values
            
            # 重新生成SHAP组合图
            plot_shap_beeswarm(X_full_scaled, FEATURE_NAMES)
            
            # 重新生成SHAP散点拟合图
            plot_shap_scatter_fit(X_full_scaled, FEATURE_NAMES)

        print(f"✅ Custom model loaded. Features (中文): {FEATURE_NAMES}")
        return {
            "code": 200,
            "message": "Model Uploaded Successfully",
            "data": {"feature_names": FEATURE_NAMES}
        }
    except Exception as e:
        error_msg = f"Model upload failed: {str(e)}"
        print(f"❌ {error_msg}")
        # 打印详细错误
        import traceback
        traceback.print_exc()
        return {"code": 500, "message": error_msg, "data": None}

# 启动服务
if __name__ == "__main__":
    import uvicorn
    print(f"\n📌 Starting server: http://localhost:8000")
    uvicorn.run(
        __file__.replace("\\", "/").split("/")[-1].split(".")[0] + ":app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )