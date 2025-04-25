import streamlit as st
import sqlite3
from hashlib import sha256
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib

# 初始化数据库
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)')

# 注册用户
def register_user(username, password):
    hashed_password = sha256(password.encode()).hexdigest()
    try:
        c.execute('INSERT INTO users VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return True
    except:
        return False

# 登录用户
def login_user(username, password):
    hashed_password = sha256(password.encode()).hexdigest()
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hashed_password))
    return c.fetchone() is not None

# 获取电影数据
def get_movie_data(keyword):
    # 连接 MySQL 数据库
    import pymysql
    try:
        db = pymysql.connect(
            host='localhost',
            user='bisheshujuku',
            password='123456',
            database='1所有电影评论'
        )
        cursor = db.cursor()
        sql = "SELECT * FROM movies WHERE title LIKE %s"
        cursor.execute(sql, (f'%{keyword}%',))
        movies = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(movies, columns=columns)
        return df
    except Exception as e:
        print(f"查询失败: {str(e)}")
        return pd.DataFrame()

# 加载模型和向量器
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.pkl')

st.title("电影评论分析系统")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

page = st.sidebar.radio("选择页面", ["登录", "注册", "数据分析", "电影查询", "评论情感预测"])

if page == "注册":
    st.header("用户注册")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("注册"):
        if register_user(username, password):
            st.success("注册成功！请返回登录。")
        else:
            st.error("用户名已存在或注册失败。")

elif page == "登录":
    st.header("用户登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        if login_user(username, password):
            st.success("登录成功！")
            st.session_state.logged_in = True
        else:
            st.error("用户名或密码错误。")

elif page == "数据分析":
    if st.session_state.logged_in:
        st.title("电影数据分析")
        # 从数据库中获取电影数据
        try:
            db = pymysql.connect(
                host='localhost',
                user='bisheshujuku',
                password='123456',
                database='1所有电影评论'
            )
            cursor = db.cursor()
            sql = "SELECT * FROM movies"
            cursor.execute(sql)
            movies = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(movies, columns=columns)

            # 电影上映年份统计
            st.subheader("电影上映年份统计")
            year_counts = df['year'].value_counts()
            st.bar_chart(year_counts)

            # 电影类型统计分析
            st.subheader("电影类型统计分析")
            genre_counts = df['genre'].str.split('/').explode().value_counts()
            st.bar_chart(genre_counts)

            # 评分分布
            st.subheader("评分分布")
            if 'rating' in df.columns:
                rating_counts = df['rating'].value_counts()
                st.bar_chart(rating_counts)

            # 电影产地
            st.subheader("电影产地")
            if 'country' in df.columns:
                country_counts = df['country'].value_counts()
                st.bar_chart(country_counts)

        except Exception as e:
            st.error(f"数据分析失败: {str(e)}")
    else:
        st.warning("请先登录。")

elif page == "电影查询":
    if st.session_state.logged_in:
        st.title("电影查询与评论分析")
        movie_name = st.text_input("请输入电影名称：")
        if st.button("查询"):
            if movie_name:
                df = get_movie_data(movie_name)
                if not df.empty:
                    st.header("电影信息")
                    st.dataframe(df[["title", "year", "genre"]])

                    # 获取电影评论
                    try:
                        db = pymysql.connect(
                            host='localhost',
                            user='bisheshujuku',
                            password='123456',
                            database='1所有电影评论'
                        )
                        cursor = db.cursor()
                        sql = "SELECT comment, sentiment FROM comments WHERE movie_id = %s"
                        cursor.execute(sql, (df['id'].values[0],))
                        comments = cursor.fetchall()
                        comment_df = pd.DataFrame(comments, columns=['comment', 'sentiment'])

                        st.header("评论情感分布")
                        sentiment_counts = comment_df['sentiment'].value_counts()
                        st.bar_chart(sentiment_counts)

                        st.header("评论关键词云图")
                        all_comments = " ".join(comment_df["comment"])
                        wordcloud = WordCloud(width=800, height=400).generate(all_comments)
                        plt.figure(figsize=(10, 5))
                        plt.imshow(wordcloud)
                        plt.axis("off")
                        st.pyplot()
                    except Exception as e:
                        st.error(f"评论分析失败: {str(e)}")
                else:
                    st.warning("未找到该电影数据。")
    else:
        st.warning("请先登录。")

elif page == "评论情感预测":
    if st.session_state.logged_in:
        st.title("用户评论情感预测")
        comment = st.text_area("请输入评论内容：")
        if st.button("预测"):
            if comment:
                # 文本预处理
                cleaned_comment = comment.lower()
                # 特征提取
                comment_vec = vectorizer.transform([cleaned_comment])
                # 情感预测
                sentiment = model.predict(comment_vec)[0]
                # 积极强度
                if hasattr(model, 'predict_proba'):
                    positive_prob = model.predict_proba(comment_vec)[:, 1][0]
                    st.write(f"情感预测：{sentiment}，积极强度：{positive_prob:.2f}")
                else:
                    st.write(f"情感预测：{sentiment}")
    else:
        st.warning("请先登录。")