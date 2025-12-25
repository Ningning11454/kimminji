

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import random
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# -------------------------- 1. 页面基础配置 --------------------------
st.set_page_config(
    page_title="学生成绩分析与预测平台",
    page_icon="📊",
    layout="wide"
)

# -------------------------- 2. 数据加载与预处理 --------------------------
@st.cache_data
def load_and_preprocess_data():
    """生成并预处理模拟学生数据"""
    # 专业和性别列表
    majors = ["大数据管理", "人工智能", "信息系统", "软件工程", "网络工程", "计算机科学"]
    genders = ["男", "女"]
    
    # 生成模拟数据
    data = []
    for i in range(300):
        major = random.choice(majors)
        gender = random.choice(genders)
        weekly_study = round(random.uniform(10, 30), 1)
        attendance = round(random.uniform(70, 100), 0)
        mid_score = round(random.uniform(60, 95), 0)
        homework = round(random.uniform(80, 100), 0)
        
        # 成绩计算逻辑（添加合理的权重）
        final_score = round(
            mid_score * 0.6 + weekly_study * 1.1 + attendance * 0.2 + homework * 0.1 + random.uniform(-3, 3),
            1
        )
        final_score = max(0, min(100, final_score))
        
        data.append({
            "学号": f"2333{random.randint(1000, 9999)}",
            "性别": gender,
            "专业": major,
            "每周学习时长(小时)": weekly_study,
            "上课出勤率": attendance,
            "期中考试分数": mid_score,
            "作业完成率": homework,
            "期末考试分数": final_score
        })
    
    df = pd.DataFrame(data)
    
    # 计算专业统计数据
    major_stats = df.groupby("专业").agg({
        "每周学习时长(小时)": "mean",
        "期中考试分数": "mean",
        "期末考试分数": "mean",
        "上课出勤率": "mean",
        "作业完成率": "mean"
    }).round(2)
    
    # 计算各专业性别比例
    gender_ratio = df.groupby(["专业", "性别"]).size().unstack(fill_value=0)
    gender_ratio["总计"] = gender_ratio.sum(axis=1)
    gender_ratio["男生比例(%)"] = (gender_ratio["男"] / gender_ratio["总计"] * 100).round(1)
    gender_ratio["女生比例(%)"] = (gender_ratio["女"] / gender_ratio["总计"] * 100).round(1)
    
    return df, major_stats, gender_ratio

# 加载数据
df, major_stats, gender_ratio = load_and_preprocess_data()

# -------------------------- 3. 机器学习模型训练 --------------------------
@st.cache_resource
def train_score_prediction_model():
    """训练成绩预测模型并返回预测函数"""
    # 特征与标签
    X = df[["性别", "专业", "每周学习时长(小时)", "上课出勤率", "期中考试分数", "作业完成率"]]
    y = df["期末考试分数"]
    
    # 类别特征编码
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    cat_features = encoder.fit_transform(X[["性别", "专业"]])
    
    # 数值特征拼接
    num_features = X[["每周学习时长(小时)", "上课出勤率", "期中考试分数", "作业完成率"]].values
    X_encoded = np.hstack([num_features, cat_features])
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    
    # 训练随机森林模型
    model = RandomForestRegressor(
        n_estimators=150, 
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    
    # 模型评估
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 定义预测函数
    def predict_score(input_data):
        """预测成绩的封装函数"""
        return model.predict(input_data)
    
    return predict_score, encoder, r2, mse, rmse

# 加载模型
predict_fn, encoder, model_r2, model_mse, model_rmse = train_score_prediction_model()

# -------------------------- 4. 页面功能函数 --------------------------
def show_project_intro():
    """项目介绍页面"""
    st.title("📚 学生成绩分析与预测平台")
    st.markdown("---")

    # 项目概述
    st.subheader("📂 项目概述")
    st.write("""
    本平台基于Streamlit框架开发，融合**数据可视化**与**机器学习**技术，
    为教育工作者和学生提供多维度的学业数据分析与个性化成绩预测服务，
    助力精准把握学习状态，提升学业表现。
    """)

    # 核心特点
    st.subheader("✨ 核心特点")
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.markdown("### 📊")
        st.write("**数据可视化**")
        st.write("多维度展示学业数据")
    with col2:
        st.markdown("### 📈")
        st.write("**专业分析**")
        st.write("对比各专业学业表现")
    with col3:
        st.markdown("### 🔮")
        st.write("**智能预测**")
        st.write("精准预测期末成绩")
    with col4:
        st.markdown("### 💡")
        st.write("**学习建议**")
        st.write("个性化提升指导")

    # 技术架构
    st.subheader("⚙️ 技术架构")
    col1, col2, col3, col4 = st.columns(4, gap="small")
    with col1:
        st.write("**前端框架**")
        st.write("Streamlit")
    with col2:
        st.write("**数据处理**")
        st.write("Pandas<br>Numpy", unsafe_allow_html=True)
    with col3:
        st.write("**可视化**")
        st.write("Plotly")
    with col4:
        st.write("**机器学习**")
        st.write("Scikit-Learn")

def show_analysis():
    """专业数据分析页面"""
    st.title("📈 专业数据分析")
    st.markdown("---")

    # 1. 各专业男女生比例分析
    st.subheader("1. 各专业男女生比例")
    fig_gender = go.Figure()
    fig_gender.add_trace(go.Bar(
        x=gender_ratio.index, y=gender_ratio["男生比例(%)"], name="男生比例",
        marker_color="#4A90E2", text=gender_ratio["男生比例(%)"].apply(lambda x: f"{x}%"),
        textposition="auto"
    ))
    fig_gender.add_trace(go.Bar(
        x=gender_ratio.index, y=gender_ratio["女生比例(%)"], name="女生比例",
        marker_color="#50E3C2", text=gender_ratio["女生比例(%)"].apply(lambda x: f"{x}%"),
        textposition="auto"
    ))
    fig_gender.update_layout(
        barmode="group", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font_color="#2C3E50", height=400, margin=dict(l=10, r=10, t=20, b=20),
        xaxis_title="专业", yaxis_title="比例(%)"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig_gender, use_container_width=True)
    with col2:
        st.write("性别比例明细")
        st.dataframe(
            gender_ratio[["男生比例(%)", "女生比例(%)"]],
            use_container_width=True,
            column_config={
                "男生比例(%)": st.column_config.NumberColumn(format="%.1f%%"),
                "女生比例(%)": st.column_config.NumberColumn(format="%.1f%%")
            }
        )

    # 2. 各专业学习时长对比
    st.subheader("2. 各专业学习时长对比")
    fig_study = px.bar(
        major_stats, x=major_stats.index, y="每周学习时长(小时)",
        color_discrete_sequence=["#E0F7FA"], height=400,
        labels={"每周学习时长(小时)": "平均学习时长(小时)", "index": "专业"}
    )
    fig_study.add_trace(go.Scatter(
        x=major_stats.index, y=major_stats["每周学习时长(小时)"],
        mode="lines+markers", name="时长趋势", line=dict(color="#FFB74D", width=3),
        marker=dict(size=8, color="#FF9800")
    ))
    fig_study.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#2C3E50",
        margin=dict(l=10, r=10, t=20, b=20)
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig_study, use_container_width=True)
    with col2:
        st.write("学习时长排名")
        st.dataframe(
            major_stats[["每周学习时长(小时)"]].sort_values(
                by="每周学习时长(小时)", ascending=False
            ),
            use_container_width=True
        )

    # 3. 各专业出勤率分析
    st.subheader("3. 各专业出勤率分析")
    fig_att = px.bar(
        major_stats, x=major_stats.index, y="上课出勤率",
        color="上课出勤率", color_continuous_scale=px.colors.sequential.YlGnBu,
        height=400, text=major_stats["上课出勤率"].apply(lambda x: f"{x}%"),
        labels={"上课出勤率": "出勤率(%)", "index": "专业"}
    )
    fig_att.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#2C3E50",
        margin=dict(l=10, r=10, t=20, b=20), coloraxis_showscale=False
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig_att, use_container_width=True)
    with col2:
        st.write("出勤率排名")
        st.dataframe(
            major_stats[["上课出勤率"]].sort_values(by="上课出勤率", ascending=False),
            use_container_width=True
        )

    # 4. 大数据管理专业专项分析
    st.subheader("4. 大数据管理专业专项分析")
    if "大数据管理" in major_stats.index:
        bigdata = major_stats.loc["大数据管理"]
        
        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4, gap="small")
        with col1:
            st.metric("平均出勤率", f"{bigdata['上课出勤率']}%")
        with col2:
            st.metric("期末平均分", f"{bigdata['期末考试分数']:.1f}分")
        with col3:
            st.metric("作业完成率", f"{bigdata['作业完成率']}%")
        with col4:
            st.metric("平均学习时长", f"{bigdata['每周学习时长(小时)']}小时")
        
        # 趋势图与成绩对比图
        col1, col2 = st.columns([2, 1])
        with col1:
            # 周出勤率趋势
            fig_big_att = go.Figure(go.Bar(
                x=["第1周", "第2周", "第3周", "第4周", "第5周"],
                y=[72, 78, 85, 88, 92],
                marker_color="#26A69A"
            ))
            fig_big_att.update_layout(
                title="大数据管理专业周出勤率趋势",
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font_color="#2C3E50", height=300,
                xaxis_title="周次", yaxis_title="出勤率(%)"
            )
            st.plotly_chart(fig_big_att, use_container_width=True)
        
        with col2:
            # 期中期末成绩对比
            fig_big_score = go.Figure(go.Bar(
                x=["期中", "期末"],
                y=[bigdata["期中考试分数"], bigdata["期末考试分数"]],
                marker_color="#81D4FA"
            ))
            fig_big_score.update_layout(
                title="成绩对比",
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font_color="#2C3E50", height=300,
                xaxis_title="考试类型", yaxis_title="分数"
            )
            st.plotly_chart(fig_big_score, use_container_width=True)
    else:
        st.warning("当前数据集中未包含「大数据管理」专业")

def show_score_prediction():
    """成绩预测页面"""
    st.title("🎯 期末成绩预测")
    st.markdown("---")

    # 提示文本
    st.write("""
    请输入学生的学习信息，系统将基于机器学习模型（随机森林回归）预测期末成绩，
    并根据预测结果提供个性化的学习建议。
    """)
    
    # 输入布局
    col1, col2 = st.columns([2, 3], gap="large")
    
    with col1:
        # 基础信息输入
        student_id = st.text_input("学号", value="23332321", placeholder="请输入学号")
        gender = st.selectbox("性别", options=["男", "女"], index=0)
        major = st.selectbox("专业", options=df["专业"].unique(), index=0)
        
        # 预测按钮
        predict_btn = st.button("预测期末成绩", type="primary", use_container_width=True)
    
    with col2:
        # 学习数据滑块
        weekly_study = st.slider(
            "每周学习时长(小时)",
            min_value=0.0, max_value=30.0, value=20.0, step=0.5,
            help="学生每周用于该课程的学习时长"
        )
        attendance = st.slider(
            "上课出勤率(%)",
            min_value=0.0, max_value=100.0, value=80.0, step=1.0,
            help="学生该课程的出勤百分比"
        )
        mid_score = st.slider(
            "期中考试分数",
            min_value=0.0, max_value=100.0, value=70.0, step=1.0,
            help="学生期中考试的分数"
        )
        homework = st.slider(
            "作业完成率(%)",
            min_value=0.0, max_value=100.0, value=90.0, step=1.0,
            help="学生作业的完成百分比"
        )
    
    # 预测逻辑
    if predict_btn:
        if not student_id.strip():
            st.error("请输入有效的学号！")
        else:
            try:
                # 特征编码
                cat_input = pd.DataFrame({"性别": [gender], "专业": [major]})
                cat_encoded = encoder.transform(cat_input)
                num_input = np.array([[weekly_study, attendance, mid_score, homework]])
                input_encoded = np.hstack([num_input, cat_encoded])
                
                # 预测成绩
                predicted_score = predict_fn(input_encoded)[0]
                predicted_score = round(max(0, min(100, predicted_score)), 1)
                
                # 结果展示
                st.markdown("---")
                st.subheader("📊 预测结果")
                
                # 成绩卡片
                st.metric(
                    label=f"学号：{student_id} - 预测期末成绩",
                    value=f"{predicted_score} 分",
                    delta=f"模型R²评分：{model_r2:.2f}",
                    delta_color="normal"
                )
                
                # 成绩等级判断和学习建议
                col_result, col_image = st.columns([2, 1])
                with col_result:
                    if predicted_score >= 90:
                        st.success("🏆 预测成绩为**优秀**，学习状态极佳！")
                        st.info("### 学习建议：\n- 保持当前的学习节奏\n- 可以尝试拓展相关知识\n- 帮助其他同学共同进步")
                    elif predicted_score >= 80:
                        st.success("🌟 预测成绩为**良好**，距离优秀仅差一步！")
                        st.info("### 学习建议：\n- 增加每周学习时长2-3小时\n- 重点复习薄弱知识点\n- 提高作业完成质量")
                    elif predicted_score >= 60:
                        st.success("🎉 预测成绩**及格**，基础达标！")
                        st.info("### 学习建议：\n- 每周至少增加5小时学习时间\n- 提高上课出勤率至90%以上\n- 及时完成并订正作业")
                    else:
                        st.warning("💪 预测成绩未及格，建议加强学习！")
                        st.info("### 学习建议：\n- 大幅增加学习时间（至少10小时/周）\n- 保证全勤上课并做好笔记\n- 寻求老师和同学的帮助\n- 制定详细的复习计划")
                
                with col_image:
                    # 根据成绩显示对应图标
                    if predicted_score >= 90:
                        st.image(
                            "https://img.icons8.com/fluency/800/000000/medal-first-place.png",
                            caption="优秀！🎉 成绩名列前茅",
                            use_container_width=True
                        )
                    elif predicted_score >= 80:
                        st.image(
                            "https://img.icons8.com/fluency/800/000000/medal-second-place.png",
                            caption="良好！💪 继续努力",
                            use_container_width=True
                        )
                    elif predicted_score >= 60:
                        st.image(
                            "https://img.icons8.com/fluency/800/000000/medal-third-place.png",
                            caption="及格！✅ 基础达标",
                            use_container_width=True
                        )
                    else:
                        st.image(
                            "https://img.icons8.com/fluency/800/000000/study.png",
                            caption="需努力！📖 加油提升",
                            use_container_width=True
                        )
                
                # 模型性能说明
                with st.expander("📈 模型性能说明", expanded=False):
                    st.write(f"模型类型：随机森林回归")
                    st.write(f"决定系数(R²)：{model_r2:.4f} (越接近1越好)")
                    st.write(f"均方误差(MSE)：{model_mse:.4f} (越小越好)")
                    st.write(f"均方根误差(RMSE)：{model_rmse:.4f} (越小越好)")
                    
            except Exception as e:
                st.error(f"预测过程中出现错误：{str(e)}")
                st.info("请检查输入数据是否有效，或刷新页面重试。")

# -------------------------- 5. 主程序 --------------------------
def main():
    """主程序入口"""
    # 侧边栏导航
    st.sidebar.title("📋 功能导航")
    st.sidebar.markdown("---")
    
    # 导航选项
    selected_page = st.sidebar.radio(
        "选择功能模块",
        options=["项目介绍", "专业数据分析", "成绩预测"],
        index=1  # 默认显示专业数据分析
    )
    
    # 模型信息
    with st.sidebar.expander("📌 模型信息", expanded=False):
        st.write(f"预测模型：随机森林回归")
        st.write(f"模型R²分数：{model_r2:.4f}")
        st.write(f"均方误差（MSE）：{model_mse:.4f}")
        st.write(f"均方根误差（RMSE）：{model_rmse:.4f}")
    
    # 页面跳转
    if selected_page == "项目介绍":
        show_project_intro()
    elif selected_page == "专业数据分析":
        show_analysis()
    elif selected_page == "成绩预测":
        show_score_prediction()
    
    # 页脚信息
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 学生成绩分析与预测平台 | 基于Streamlit开发")

# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    main()
