import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from functools import reduce
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    matthews_corrcoef
)

st.set_page_config(layout='wide', page_title='データ分析EDA', page_icon='📊')

# 属性が存在しない場合に初期化
keys_defaults = [
    ('df_train', None),
    ('df_test', None),
    ('target', None),
    ('features', None),
    ('default_df_train', None),
    ('default_df_test', None),
    ('default_target', None),
    ('default_features', None)
]

# ループを使ってセッションステートを初期化
for key, default_value in keys_defaults:
    if key not in st.session_state:
        st.session_state[key] = default_value

# datasetの初期化
if st.session_state.df_train is not None and st.session_state.df_test is not None:
    df_train = st.session_state.df_train
    df_test = st.session_state.df_test
    target = st.session_state.target
    features = st.session_state.features
else:
    df_train = pd.read_csv('dataset/titanic_data/titanic_train.csv')
    df_test = pd.read_csv('dataset/titanic_data/titanic_test.csv')
    target = 'Survived'
    df_features = df_train.drop(['PassengerId', 'Survived'], axis=1)
    features = df_features.columns.tolist()
    st.session_state.default_df_train = df_train
    st.session_state.default_df_test = df_test
    st.session_state.default_target = target
    st.session_state.default_features = features

# dataset関数
def dataset():
    if st.session_state.df_train is not None:
        st.sidebar.info('ユーザーのTrainDataを使用中')
    else:
        st.sidebar.warning('デフォルトのTrainDataを使用中')

    if st.session_state.df_test is not None:
        st.sidebar.info('ユーザーのTestDataを使用中')
    else:
        st.sidebar.warning("デフォルトのTestDataを使用中")

    if st.session_state.target is not None:
        st.sidebar.info(f'目的変数: {st.session_state.target}')
    else:
        st.sidebar.warning(f'目的変数: "Survived"')
        
    st.sidebar.markdown('---')

# サイドバー設定
st.sidebar.title('データ分析EDA (分類)')

home = 'ホーム'
loading_data = 'データの準備'
data_summary = 'データ概要'
each_feature = '各特徴量の要約'
model = 'モデル構築'

selection = [home, loading_data, data_summary, each_feature, model]
choice = st.sidebar.selectbox('メニュー', selection, key='menu')

# ホーム---------------------
if choice == home:
    home_tab1, home_tab2 = st.tabs(['アプリ概要','ガイド'])

    #アプリ概要
    with home_tab1:
        st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
        st.image("img/EDA.jpg", width=int(1200 * 0.2))
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '''
                <div style="text-align: left;"><h2 style="font-size: 24px;">アプリ概要</h2>
                このアプリはテーブルデータの分類タスクにおけるEDAを助けます。<br>
                データの概要・各特徴量の把握、特徴量同士の関係性を表・グラフ化します。<br>
                また、LGBMでの単純なモデルの構築ができ、学習・予測を経て簡単なレポートを作成します。<br>
                <br>
                <div style="text-align: right;">2024.8.14</div><br>
                </div>
                ''', unsafe_allow_html=True)
            
        st.info('当アプリはKaggleで提供されているデータセットを想定して作成しました。')

    # ガイド
    with home_tab2:        
        slides = [f'img/guide/slide{i}.JPG' for i in range(1, 8)]
        for slide in slides:
            st.image(slide, caption='', use_column_width=True)

# データの準備--------------
if choice == loading_data:

    dataset()

    st.markdown('<h2 style="font-size: 24px;">データのアップロード</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:

        # データの読み込み割合を設定できるスライダー
        sample_ratio = st.slider(
            'データの読み込み割合を設定 (%)', 
            min_value=10, max_value=100, step=10, value=100
        ) / 100  # パーセントを小数に変換

        st.markdown('**学習用のデータをアップロード**')
        uploaded_train_file = st.file_uploader('学習用データ(以降TrainData)', type=['csv'])
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('**予測用のデータをアップロード**')
        uploaded_test_file = st.file_uploader('予測用データ(以降TestData)', type=['csv'])
        st.markdown('<br>', unsafe_allow_html=True)
    
    with col2:
        # 学習用データの処理
        if uploaded_train_file is not None:
            df_train = pd.read_csv(uploaded_train_file)
            
            # データを指定された割合でサンプリング
            if sample_ratio < 1.0:
                df_train = df_train.sample(frac=sample_ratio, random_state=42)
            
            st.info(f'トレーニングデータがアップロードされました👌 (設定読込割合:{int(sample_ratio * 100)}%)')
            st.session_state.df_train = df_train
        else:
            st.warning('現在デフォルトのトレーニングデータを使用しています。')

        # 検証用データの処理
        if uploaded_test_file is not None:
            df_test = pd.read_csv(uploaded_test_file)
        
            # データを指定された割合でサンプリング
            if sample_ratio < 1.0:
                df_test = df_test.sample(frac=sample_ratio, random_state=42)
            
            st.info(f'テストデータがアップロードされました👌 (設定読込割合:{int(sample_ratio * 100)}%)')
            st.session_state.df_test = df_test
            
        else:
            st.warning('現在デフォルトのテストデータを使用しています。')

        # 目的変数と説明変数の設定
        if uploaded_train_file is not None:
            st.error('目的変数・説明変数を指定してください。')
            
            feature_target = df_train.columns.tolist()
            # 目的変数の選択
            target = st.selectbox('目的変数', feature_target, index=0)
            
            features_drop_target = [feature for feature in feature_target if feature != target]
            # 説明変数の選択
            features = st.multiselect('説明変数', features_drop_target, features_drop_target)

            st.session_state.target = target
            st.session_state.features = features
            
        else:
            st.warning('目的変数はデフォルトです。')
            


#データ概要------------------------------------------
if choice == data_summary:
    dataset()

    tab1, tab2, tab3, tab4 = st.tabs(['データの確認', 'データの要約', 'ヒートマップ', 'ペアプロット']) 

    #データの確認
    with tab1:
        st.markdown('<h2 style="font-size: 24px;">データの確認</h2>', unsafe_allow_html=True)
        st.markdown('<h3 style="font-size: 20px;">TrainData</h3>', unsafe_allow_html=True)
        if len(df_train) > 1000:
            st.write('1000行を超えるデータのため最初の100行のみ表示します。')
            st.dataframe(df_train.head(100))
        else:
            st.dataframe(df_train)

        st.markdown('---')
        
        st.markdown('<h3 style="font-size: 20px;">TestData</h3>', unsafe_allow_html=True)
        if len(df_test) > 1000:
            st.write('1000行を超えるデータのため最初の100行のみ表示します。')
            st.dataframe(df_test.head(100))
        else:
            st.dataframe(df_test)

    #データの要約
    with tab2:
        st.markdown('<h2 style="font-size: 24px;">データの要約</h2>', unsafe_allow_html=True)
        tab2_col1, tab2_col2 = st.columns(2)
        
        def dataset_overview(df):
            column_names = df.columns
            data_types = df.dtypes
            missing_values = df.isnull().sum()
            total_rows = len(df)
            missing_percent = (missing_values / total_rows) * 100
            unique_counts = [df[col].nunique() for col in column_names]

            summary = pd.DataFrame({
                'Dtype': data_types,
                '欠損値': missing_values,
                '欠損値割合(%)': np.floor(missing_percent).astype(int),
                'ユニークな値の数': unique_counts
            })
            return st.dataframe(summary)
        
        with tab2_col1:
            st.markdown('<h3 style="font-size: 20px;">TrainData</h3>', unsafe_allow_html=True)
            st.write(f'{df_train.shape[0]}行 × {df_train.shape[1]}列')
            dataset_overview(df_train)

        with tab2_col2:
            st.markdown('<h3 style="font-size: 20px;">TestData</h3>', unsafe_allow_html=True)
            st.write(f'{df_test.shape[0]}行 × {df_test.shape[1]}列')
            dataset_overview(df_test)

    with tab3:
        original_labels = {col: df_train[col].unique() for col in df_train.columns if df_train[col].dtype == 'object'}

        # データのエンコード
        label_encoders = {}
        encoded_df = df_train.copy()
        for col in encoded_df.columns:
            if encoded_df[col].dtype == 'object':
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(encoded_df[col])
                label_encoders[col] = le

        options = st.multiselect(
            '特徴量の選択',
            [target] + features,
            [target] + features,
            key='heatmap_select_features'
        )

        st.header("ヒートマップ")
        if st.button('実行', key='unique_key_1'):
            with st.spinner('Generating heatmap...'):
                if not options:
                    st.write('選択された特徴量がありません。')
                else:
                    corr_matrix = encoded_df[options].corr()
                    tab1_col1, tab1_col2 = st.columns(2)
                    
                    with tab1_col1:
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(corr_matrix, annot=None, cmap='coolwarm', center=0,
                                    xticklabels=corr_matrix.columns, yticklabels=corr_matrix.index)
                        st.pyplot(plt)
                        plt.close() 

                    with tab1_col2:
                        st.dataframe(corr_matrix)
                        
                    corr_matrix_abs = corr_matrix.abs()
                    mask = np.triu(np.ones_like(corr_matrix_abs, dtype=bool))
                    tri_corr_matrix = corr_matrix_abs.where(mask)
                    np.fill_diagonal(tri_corr_matrix.values, np.nan)
                    sorted_corr = tri_corr_matrix.unstack().sort_values(ascending=False)
                    top_10_corr = sorted_corr.head(10)
                    
                    st.write("### Top 10")
                    st.dataframe(top_10_corr.reset_index(name='Correlation').rename(columns={'level_0': 'Feature1', 'level_1': 'Feature2'}))


    with tab4:
        st.header('ペアプロット')

        st.write('図が見にくい場合,動作が重い場合は特徴量を減らしてみてください。')

        options = st.multiselect(
            '特徴量の選択',
            features,
            features,
            key='pairplot_select_features'
        )

        if st.button('実行', key='unique_key_2'):
            with st.spinner('Generating graph...'):
                if len(options) < 2:
                    st.write('少なくとも２つ以上の選択肢を選んでください。')
                else:
                    df_subset = df_train[options].copy()

                    # ラベルエンコーディング
                    le = LabelEncoder()
                    for col in options:
                        if df_subset[col].dtype == 'object':  # カテゴリ変数のみをエンコード
                            df_subset[col] = le.fit_transform(df_subset[col])

                    # 目的変数を`df_subset`に追加
                    df_subset[target] = df_train[target]

                    # ペアプロットの生成
                    pairplot = sns.pairplot(df_subset, hue=target, palette='viridis', plot_kws={'alpha': 0.6}, corner=True)
                    st.pyplot(pairplot.figure)
                    plt.close()




# 各特徴量の要約---------------------
if choice == each_feature:

    dataset()
    feature_choice = st.sidebar.selectbox('特徴量の選択', [target] + features, index=0)

    # ユニークな値の取得
    def get_unique_values(df, column):
        if column in df.columns:
            return df[column].unique()
        else:
            return []
    #　ユニークな値の表示    
    def display_values_with_quotes(values):
        quoted_values = [f"'{value}'" for value in values]
        return ', '.join(quoted_values)
    # ヒストグラム
    def plot_histogram_kde(df, feature_choice):
        bins = st.number_input('ヒストグラムのビンの数を設定', min_value=5, max_value=1000, value=30, step=1, key=f'{df}_bins_{feature_choice}')
        with st.spinner('Generating graph...'):
            plt.figure(figsize=(10, 6))
            data = df[feature_choice].dropna()

            # ヒストグラム
            ax1 = plt.gca()
            ax1.hist(data, bins=bins, density=False, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density', color='skyblue')
            ax1.tick_params(axis='y', labelcolor='skyblue')

            # KDE
            ax2 = ax1.twinx()
            kde = gaussian_kde(data, bw_method=0.3)
            x = np.linspace(data.min(), data.max(), 1000)
            ax2.plot(x, kde(x), color='red', label='KDE')
            ax2.set_ylabel('Density', color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # 重ねて表示
            plt.title(f'{feature_choice} Histogram + KDE')
            ax1.grid(True)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            st.pyplot(plt)
            plt.clf()
    # 箱ひげ図
    def plot_boxplot(df, feature_choice):
        plt.figure(figsize=(10, 6))
        plt.boxplot(df[feature_choice].dropna(), vert=False, patch_artist=True, 
                    boxprops=dict(facecolor='skyblue', color='black'), 
                    whiskerprops=dict(color='black'), 
                    medianprops=dict(color='red'))
        plt.title(f'{feature_choice} Box Plot')
        plt.xlabel('Value')
        plt.grid(True)
        
        st.pyplot(plt)
        plt.clf()
    # 散布図
    def plot_scatter(df, feature_choice, target):
        plt.figure(figsize=(10, 6))
        plt.scatter(df[target], df[feature_choice], color='skyblue')
        plt.title(f'Scatter Plot of {target} and {feature_choice}')
        plt.xlabel(target)
        plt.ylabel(feature_choice)
        plt.grid(True)
        
        st.pyplot(plt)
        plt.clf() 
    # 棒グラフ
    def plot_bar_chart(df, feature_choice, chart_key):
        value_counts = df[feature_choice].value_counts().sort_values(ascending=False).reset_index()
        value_counts.columns = [feature_choice, 'Count']
        max_items = len(value_counts)
        num_items = st.slider(
            "表示する範囲を指定",
            min_value=1,
            max_value=max_items,
            value=min(10, max_items),
            key=f"slider_for_bar_chart_{chart_key}"
        )

        top_items = value_counts.head(num_items)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_choice, y='Count', data=top_items, palette='Blues_r')
        plt.title(f"Bar Chart of {feature_choice}")
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')

        st.pyplot(plt)
        plt.clf()
    # クラス割合棒グラフ
    def plot_target_ratio(df, feature_choice, target, chart_key):

        value_counts = df[feature_choice].value_counts().sort_values(ascending=False).reset_index()
        value_counts.columns = [feature_choice, 'Count']

        max_items = len(value_counts)
        num_items = st.slider(
            "表示する範囲を指定",
            min_value=1,
            max_value=max_items,
            value=min(10, max_items),
            key=f"slider_for_bar_chart_{chart_key}"
        )
        
        top_items = value_counts.head(num_items)[feature_choice].tolist()
        cross_table = pd.crosstab(df[feature_choice], df[target], normalize='index')
        
        top_items = [item for item in top_items if item in cross_table.index]
        sorted_cross_table = cross_table.loc[top_items]

        # ターゲットのユニークなクラス数に応じて色を設定
        unique_classes = df[target].nunique()
        palette = sns.color_palette('Set3', unique_classes)  # 'husl' パレットは色の区別がはっきりする

        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_cross_table.plot(kind='bar', stacked=True, ax=ax, color=palette)

        ax.set_title(f'{feature_choice} by {target} Ratio')
        ax.set_xlabel(f'{feature_choice}')
        ax.set_ylabel('Ratio')
        ax.set_xticks(range(len(sorted_cross_table.index)))
        ax.set_xticklabels(sorted_cross_table.index, rotation=45)
        ax.legend(title=target, loc='upper right')
        ax.grid(axis='y')

        st.pyplot(fig)
        plt.clf()
    #バイオリンプロット
    def violin_plot(df, feature, chart_key):
        value_counts = df[feature].value_counts()

        max_items = len(value_counts)
        num_items = st.slider(
            '表示する範囲を指定',
            min_value=1,
            max_value=max_items,
            value=min(10, max_items),
            key=f'slider_for_violin_plot_{chart_key}'
        )

        sorted_values_by_frequency = value_counts.sort_values(ascending=False).head(num_items).index
        filtered_df = df[df[feature].isin(sorted_values_by_frequency)]

        plt.figure(figsize=(10, 6))

        if df[feature].dtype == 'object':
            sns.violinplot(x=feature, y=target, data=filtered_df)
            plt.title(f'{feature} vs {target}')
        else:
            sns.violinplot(x=target, y=feature, data=filtered_df)
            plt.title(f'{feature} vs {target}')

        plt.xticks(rotation=45)
        st.pyplot(plt)
        plt.clf()

    #量的変数の処理
    def numeric_feature(df_train, df_test, feature_choice):
        st.write(f'<h1 style="font-size: 24px;">{feature_choice} の要約</h1>', unsafe_allow_html=True)
        st.markdown(
            f'**{feature_choice}**を**量的変数**と判別しました。<br>'
            f'Dtypeは`{df_train[feature_choice].dtype}`です。<br>'
            f'欠損値は**{df_train[feature_choice].isnull().sum()}**個あります。<br>',
            unsafe_allow_html=True
        )

        if feature_choice == target:
            plot_histogram_kde(df_train, feature_choice)
            st.markdown('<h3 style="font-size: 20px;">要約統計量</h3>', unsafe_allow_html=True)
            st.dataframe(df_train[feature_choice].describe())
        
        else:
            st.markdown('<h3 style="font-size: 20px;">TrainDataとTestDataの比較</h3>', unsafe_allow_html=True)

            con_hist_col1, con_hist_col2 = st.columns(2)
            with con_hist_col1:
                st.markdown('**TrainData**')
                plot_histogram_kde(df_train, feature_choice)
            with con_hist_col2:
                st.markdown('**TestData**')
                plot_histogram_kde(df_test, feature_choice)

            st.markdown('<h3 style="font-size: 20px;">目的変数との関係</h3>', unsafe_allow_html=True)

            vs_target_col1, vs_target_col2 = st.columns(2)
            with vs_target_col1:
                plt.figure(figsize=(10, 6))
                sns.violinplot(x=target, y=feature_choice, data=df_train)
                plt.title(f'{feature_choice} vs {target}')
                plt.xticks(rotation=0)
                st.pyplot(plt)
                plt.clf()
            with vs_target_col2:
                plot_scatter(df_train, feature_choice, target)

            st.markdown('<h3 style="font-size: 20px;">要約統計量</h3>', unsafe_allow_html=True)
            st.dataframe(df_train[feature_choice].describe())

    #質的変数の処理
    def categorical_feature(df_train, df_test, feature_choice, max_display=10):
        st.write(f'<h1 style="font-size: 24px;">{feature_choice} の要約</h1>', unsafe_allow_html=True)

        unique_values = df_train[feature_choice].unique()
        num_unique_values = len(unique_values)

        if num_unique_values > 20:
            # ユニークな値が20個を超える場合
            displayed_values = list(map(str, unique_values[:max_display]))  # 最初のmax_display個を表示
            truncated_message = f"（... 他 {num_unique_values - max_display} 個の値があります）"
            formatted_values = "', '".join(displayed_values) + truncated_message
        else:
            # ユニークな値が20個以下の場合
            formatted_values = "', '".join(map(str, unique_values))

        st.markdown(
            f'''
            **{feature_choice}**を**質的変数**と判別しました。<br>
            Dtype : `{df_train[feature_choice].dtype}`<br>
            欠損値 : **{df_train[feature_choice].isnull().sum()}**個<br>
            ユニークな値({num_unique_values}個) : <br>['{formatted_values}']<br><br>
            ''',
            unsafe_allow_html=True
        )

        if feature_choice == target:
            plot_bar_chart(df_train, feature_choice, '1')
        else:
            st.markdown('<h3 style="font-size: 20px;">TrainDataとTestDataの比較</h3>', unsafe_allow_html=True)
            
            cat_plot_col1, cat_plot_col2 = st.columns(2)
            with cat_plot_col1:
                plot_bar_chart(df_train, feature_choice, '2')
            with cat_plot_col2:
                plot_bar_chart(df_test, feature_choice, '3')
            
            st.markdown('<h3 style="font-size: 20px;">目的変数との関係</h3>', unsafe_allow_html=True)
            vs_target_col1, vs_target_col2 = st.columns(2)
            with vs_target_col1:
                plot_target_ratio(df_train, feature_choice, target, '4')  # target を追加
            
            st.markdown('<h3 style="font-size: 20px;">ユニークな値の数</h3>', unsafe_allow_html=True)
            
            value_counts = df_train[feature_choice].value_counts()
            value_counts_df = value_counts.reset_index()
            value_counts_df.columns = ['Unique', 'UniqueCount']
            st.dataframe(value_counts_df)

    #時系列変数の処理
    def datatime_feature():
        st.write(f'<h1 style="font-size: 24px;">{feature_choice} の時系列要約</h1>', unsafe_allow_html=True)

    #ユーザーが任意の変数尺度に変更した場合の処理
    def select_type(index_num):
        select_type = st.sidebar.radio('変数尺度の変更', ['量的変数', '質的変数'], index=index_num)
        if select_type == '量的変数':
            numeric_feature(df_train, df_test, feature_choice)
        elif select_type == '質的変数':
            categorical_feature(df_train, df_test, feature_choice)

    # 欠損値を除外したデータを取得
    cleaned_data = df_train[feature_choice].dropna()

    # 量的変数の条件
    if pd.api.types.is_numeric_dtype(cleaned_data):
        # 数値型でユニークな値が100未満の場合はカテゴリカルとみなす
        if cleaned_data.nunique() < 50:
            select_type(1)  # カテゴリカルなデータとして処理
        else:
            select_type(0)  # 連続的な数値データとして処理

    # 質的変数の条件
    elif (
            pd.api.types.is_categorical_dtype(cleaned_data) or
            pd.api.types.is_string_dtype(cleaned_data) or
            cleaned_data.dtype == 'object'
        ):
        select_type(1)  # カテゴリカルなデータとして処理

    # 時系列変数の条件
    elif pd.api.types.is_datetime64_any_dtype(cleaned_data):
        select_type(2)  # 時系列データとして処理

#モデル構築--------------------------
elif choice == model:

    dataset()
    df_sample = df_train.copy()
    st.write('LGBMの基本的なモデル構築・学習・検証をし、レポートを作成します。')
    
    st.header("モデル構築・学習・予測")

    st.markdown('---')

    # 使用するデータの割合と検証データの割合を選択
    data_usage_rate = st.slider('使用するTrainDataの割合を設定してください。', 0.0, 1.0, 1.0)
    data_usage_testrate = st.slider('TrainDataの検証データ割合を設定してください。', 0.0, 1.0, 0.3)

    st.markdown('---')



    # 目的変数の選択
    target = st.selectbox(
        '目的変数の確認',
        [target]+features,
        index=0
    )

    # 目的変数選択後に特徴量リストを更新
    feature = [col for col in df_sample.columns if col != target]

    # マルチセレクトボックスで使用する特徴量を選択
    model_feature_select = st.multiselect(
        '説明変数を指定してください。',
        options=features,
        default=features
    )

    st.markdown('---')

    purpose1 = '全体的な正解率を評価したい'
    purpose2 = 'モデルの全体的なバランスを評価したい'
    purpose3 = '偽陽性を減らしたい'
    purpose4 = '偽陰性を減らしたい'
    purpose5 = 'モデルの確率予測の正確さを評価したい'

    classification_select = st.selectbox(
        '何を目的としますか？',
        [purpose1, purpose2, purpose5, purpose3, purpose4,]
    )
    
    value_counts = df_sample[target].value_counts(normalize=True)

    # 比率を整数比に変換する関数
    def compute_ratio(ratios):
        # 比率を整数に変換
        scale_factor = 1 / min(ratios)  # 最小値でスケーリング
        int_ratios = [round(ratio * scale_factor) for ratio in ratios]
        # 最大公約数で割る
        def find_gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        ratio_gcd = reduce(find_gcd, int_ratios)
        return [r // ratio_gcd for r in int_ratios]

    # 比率を計算
    ratios = value_counts.values
    ratios_int = compute_ratio(ratios)
    labels = [f'{label}' for label in value_counts.index]

    # 比率の差を判断するための関数
    def categorize_ratio(ratios):
        max_ratio = max(ratios)
        min_ratio = min(ratios)
        ratio_diff = max_ratio / min_ratio
        
        if ratio_diff >= 10:
            return '高い比率差'
        elif ratio_diff >= 5:
            return '中程度の比率差'
        elif 2<= ratio_diff < 5:
            return '低い比率差'
        else:
            return 'ほぼ均衡'
        
    # 比率のカテゴリーを決定
    ratio_category = categorize_ratio(ratios)

    # 比率をフォーマット
    class_ratio = (f"クラス比率は {'：'.join(labels)} = {'：'.join(map(str, ratios_int))}で{ratio_category}")
    st.session_state.class_ratio = class_ratio

    if ratio_category == '高い比率差':
        additional_metrics = ['ROC-AUC', 'F1', 'MCC']
        st.info(f'`{class_ratio}です。データ量と比較してクラス不均衡がみられます。評価指標ROC-AUC,F1,MCCを考慮してください。`')
    else:
        additional_metrics = []
        st.write(f'`{class_ratio}です。(データ量と比較してクラス不均衡が気になる場合評価指標ROC-AUC,F1,MCCを考慮してください。)`')

    st.write('おすすめの評価指標')

    if classification_select == purpose1:
        eval_metric_default = ['Accuracy']
        
    elif classification_select == purpose2:
        eval_metric_default = ['F1', 'MCC']
        
    elif classification_select == purpose3:
        eval_metric_default = ['Precision']
        
    elif classification_select == purpose4:
        eval_metric_default = ['Recall']
    
    else:
        eval_metric_default = ['LogLoss']

    # 追加の評価指標を加える
    eval_metric_default.extend(additional_metrics)

    # 結果を表示
    if len(eval_metric_default) == 1:
        st.write(eval_metric_default[0])
    else:
        st.write(', '.join(eval_metric_default))

    eval_metric = st.multiselect(
        '評価指標を選んでください。',
        options=['Accuracy', 'Precision', 'Recall', 'F1', 'LogLoss','ROC-AUC', 'MCC'],
        default=eval_metric_default
    )
    st.session_state.classification_eval_metric = eval_metric

    unique_classes = df_sample[target].unique()
    target_label_encoders = {}

    st.markdown('---')
    
    if len(unique_classes) == 2:
        # 2クラスの場合、陽性ラベルを選択させる
        positive_label = st.radio('陽性ラベルを選んでください', options=unique_classes, key='positive_label_radio')
        negative_label = [cls for cls in unique_classes if cls != positive_label][0]
        
        # ラベルエンコーダを作成して保存
        le = LabelEncoder()
        le.fit([negative_label, positive_label])
        target_label_encoders[target] = le
        # ラベルのエンコード
        df_sample[target] = df_sample[target].map({negative_label: 0, positive_label: 1})
        
    elif len(unique_classes) > 2:
        st.write('多クラス分類のため自動でラベルを割り当てます。')

    else:
        st.warning("クラスが1つしかありません。")

    st.markdown('---')

    if st.button('学習・評価'):

        with st.spinner('Learning...'):
            
            # 選択されたデータの準備
            df_sample = df_sample.sample(frac=data_usage_rate, random_state=42)
            st.session_state.df_sample = df_sample
            
            # カテゴリカルデータのエンコード(target以外)
            label_encoders = {}
            for col in df_sample.columns:
                if df_sample[col].dtype == 'object' and col != target:
                    le = LabelEncoder()
                    df_sample[col] = le.fit_transform(df_sample[col])
                    label_encoders[col] = le
                                
            # 特徴量とターゲットを準備
            X = df_sample[model_feature_select]
            y = df_sample[target]

            # データの分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_usage_testrate, random_state=42)

            # モデルの訓練
            model = lgb.LGBMClassifier()  # 分類タスクに対応するモデル
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 逆変換を適用する（必要に応じて）
            if target in target_label_encoders:
                le = target_label_encoders[target]
                y_test = le.inverse_transform(y_test)
                y_pred = le.inverse_transform(y_pred)
            # classification_reportを作成
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()

            #Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            #Precision
            def precision(average):
                precision_ = {}
                precision_[average] = precision_score(y_test, y_pred, average=average)
                st.session_state[f'classification_precision_{average}'] = precision_[average]
                return precision_[average]
            precision_micro = precision('micro')
            precision_macro = precision('macro')
            precision_weighted = precision('weighted')
            precision = f"Precision || M**i**croAvg: {precision_micro:.4f}  |  M**a**croAvg: {precision_macro:.4f}  |  WeightedAvg: {precision_weighted:.4f}"
            #Recall
            def recall(average):
                recall_ = {}
                recall_[average] = recall_score(y_test, y_pred, average=average)
                st.session_state[f'classification_recall_{average}'] = recall_[average]
                return recall_[average]
            recall_micro = recall('micro')
            recall_macro = recall('macro')
            recall_weighted = recall('weighted')
            recall = f"Recall || M**i**croAvg: {recall_micro:.4f}  |  M**a**croAvg: {recall_macro:.4f}  |  WeightedAvg: {recall_weighted:.4f}"
            #F1
            def f1(average):
                f1_ = {}
                f1_[average] = f1_score(y_test, y_pred, average=average)
                st.session_state[f'classification_f1_{average}'] = f1_[average]
                return f1_[average]
            f1_micro = f1('micro')
            f1_macro = f1('macro')
            f1_weighted = f1('weighted')
            f1 = f"F1 || M**i**croAvg: {f1_micro:.4f}  |  M**a**croAvg: {f1_macro:.4f}  |  WeightedAvg: {f1_weighted:.4f}"
            #LogLoss
            y_proba = model.predict_proba(X_test)
            loss = log_loss(y_test, y_proba)
            #MCC
            mcc = matthews_corrcoef(y_test, y_pred)
            #ROC-AUC
            if len(unique_classes) == 2:
                positive_label = list(target_label_encoders[target].classes_)[1]
                negative_label = list(target_label_encoders[target].classes_)[0]
                y_test_binary = np.where(y_test == positive_label, 1, 0)
                y_proba = model.predict_proba(X_test)[:, 1] 
                roc_auc = roc_auc_score(y_test_binary, y_proba)
            elif len(unique_classes) > 2:
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

            def aprf(evaluation):
                if 0.9 <= evaluation:
                    st.markdown('`90%以上は非常に良い性能を示しています。`')
                elif 0.7 < evaluation < 0.9:
                    st.markdown('`70%~90%は比較的良い性能を示しています。`')
                else:
                    st.markdown('`70%以下はあまり良い性能とは言えません。`')

            st.markdown(f'<h2 style="font-size: 24px;">評価</h2>', unsafe_allow_html=True)
            if 'Accuracy' in eval_metric:
                st.info(f'✅ Accuracy: {accuracy:.4f}')
                st.write('Acuuracy(正解率)は全体の予測がどれだけ正確か示します。')
                aprf(accuracy)
            if 'Precision' in eval_metric:
                st.info('✅ ' + precision)
                st.write('Precision(適合率)は陽性と予測された中で実際に陽性である割合を示します。偽陽性を減らしたい場合に重要です。')
                aprf(precision_weighted)
            if 'Recall' in eval_metric:
                st.info('✅ ' + recall)
                st.write('Recall(再現率)は実際の陽性の中で正しく陽性と予測された割合を示します。偽陰性を減らしたい場合に重要です。')
                aprf(recall_weighted)
            if 'F1' in eval_metric:
                st.info('✅ ' + f1)
                st.write('F1はPrecisionとRecallの調和平均です。両者のバランスを取ります。')
                aprf(f1_weighted)
            if 'LogLoss' in eval_metric:
                st.info(f'✅ Log Loss: {loss:.4f}')
                st.write('LogLossはモデルが予測した確率と実際のラベルとの間の距離を測る指標です。小さいほど良いです。')
                if 0.1 > loss:
                    st.markdown('`0.1以下は非常に良い性能を示しています。`')
                elif 0.1<= loss <= 0.5:
                    st.markdown('`0.1~0.5は比較的良い性能を示しています。`')
                else:
                    st.markdown('`0.5以上はあまり良い性能とは言えません。`')
            if 'MCC' in eval_metric:
                st.info(f'✅ MCC: {mcc:.4f}')
                st.write('MCC(マシューズ相関係数)は分類結果の全体的な品質を示す指標です。-1から1の範囲で、1は完璧な分類を示します。')
                aprf(mcc)
            if 'ROC-AUC' in eval_metric:
                if len(unique_classes) == 2:
                    st.info(f'✅ ROC-AUC: {roc_auc:.4f}')
                    st.write('ROC曲線の下の面積です。クラスの分離能力を示します。')
                    aprf(roc_auc)
                elif len(unique_classes) > 2:
                    st.info(f'✅ ROC-AUC (One-vs-Rest): {roc_auc:.4f}')
                    st.write('ROC曲線の下の面積です。クラスの分離能力を示します。')
                    aprf(roc_auc)                 

            st.markdown(f'<h2 style="font-size: 24px;">レポート</h2>', unsafe_allow_html=True)
            st.markdown(f'**概要**', unsafe_allow_html=True)
            st.markdown(
                f"""
                    {df_train.shape[0]}行のデータを{int(data_usage_rate * 100)}%、その内{int(data_usage_testrate * 100)}%を検証用データとして使用しました。<br>
                    目的変数は {target}。ユニークな値は{df_train[target].unique()}で、{class_ratio}です。<br>
                    説明変数は以下の{len(model_feature_select)}個です。<br>
                    <br>
                    {model_feature_select}<br>
                    <br>
                    LGBMClassifierを使用して分類モデルを作成。パラメータはデフォルト値を使用しています。<br>
                    LGBMを使用するにあたり、文字列をscikit-learnのLabelEncoderでエンコーディングしました。<br>
                    欠損値、外れ値、異常値の処理は行っていません。<br>
                    <br>
                """, unsafe_allow_html=True)
            
            st.markdown(f'**評価結果**', unsafe_allow_html=True)
            st.write(f'Accuracy: {accuracy:.4f}')
            st.markdown(precision)
            st.markdown(recall)
            st.markdown(f1)
            st.write(f'Log Loss: {loss:.4f}')
            st.write(f'MCC: {mcc:.4f}')
            if len(unique_classes) == 2:
                st.write(f'ROC-AUC: {roc_auc:.4f}')
            elif len(unique_classes) > 2:
                st.write(f'ROC-AUC (One-vs-Rest): {roc_auc:.4f}')
            
            st.markdown(f'<br>**混同行列 / 予測確立 / ROC曲線**', unsafe_allow_html=True)                
            report1_col1, report1_col2, report1_col3 = st.columns(3)
            with report1_col1:
                def cm():
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))  # サイズを指定
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_classes, yticklabels=unique_classes)
                    plt.xlabel('Predicted labels')
                    plt.ylabel('True labels')
                    plt.title('Confusion Matrix')
                    st.pyplot(plt)
                    plt.clf()

                cm()

            with report1_col2:
                if len(unique_classes) == 2:
                    # 二値分類の場合のヒストグラム
                    plt.figure(figsize=(10, 6))
                    plt.hist(y_proba, bins=20, color='skyblue', edgecolor='black')
                    plt.xlabel('Predicted Probability')
                    plt.ylabel('Frequency')
                    plt.title('Histogram of Predicted Probabilities')
                    st.pyplot(plt)
                    plt.clf()

                if len(unique_classes) > 2:
                    plt.figure(figsize=(10, 6))
                    df_proba = pd.DataFrame(y_proba, columns=[f'{cls}' for cls in unique_classes])
                    df_proba_melted = df_proba.melt(var_name='Class', value_name='Predicted Probability')
                    
                    sns.violinplot(x='Class', y='Predicted Probability', data=df_proba_melted, palette='viridis')
                    
                    plt.xlabel('Class')
                    plt.xticks(rotation=90)
                    plt.ylabel('Predicted Probability')
                    plt.title('Violin Plot of Predicted Probabilities by Class')
                    st.pyplot(plt)
                    plt.clf()
                
            with report1_col3:
                if len(unique_classes) == 2:
                    positive_label = list(target_label_encoders[target].classes_)[1]
                    y_test_binary = np.where(y_test == positive_label, 1, 0)
                    y_proba = model.predict_proba(X_test)[:, 1]
                    
                    # ROC曲線の計算
                    fpr, tpr, _ = roc_curve(y_test_binary, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # ROC曲線のプロット
                    plt.figure(figsize=(10, 7))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic')
                    plt.legend(loc='lower right')
                    st.pyplot(plt)
                    plt.clf()
                
                # 多クラス分類の場合
                if len(unique_classes) > 2:
                    y_test_bin = label_binarize(y_test, classes=unique_classes)
                    y_proba = model.predict_proba(X_test)
                    
                    plt.figure(figsize=(10, 7))
                    for i in range(len(unique_classes)):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(unique_classes[i], roc_auc))

                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic - Multi-class')
                    plt.legend(loc='lower right')
                    st.pyplot(plt)
                    plt.clf()

            st.markdown(
                f"""
                ご自身のタスク、他の指標と比べて性能を評価してください。<br>
                パラメータチューニングを行うことでさらに性能が上がる可能性があります。<br>
                評価したい指標が高い場合でも、交差検証を行い性能の安定性を確認してください。<br>
                評価したい指標が低い場合、LGBM以外のモデルを検討してください。<br>
                """, unsafe_allow_html=True)
            
            st.markdown(f'<br>**特徴量重要度 / 相関係数TOP5**', unsafe_allow_html=True)
            
            feature_importance_col1, feature_importance_col2 = st.columns(2)
            with feature_importance_col1:
                # 特徴量重要度の取得と表示
                feature_importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': model_feature_select,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)
                
                st.write("特徴量重要度")
                plt.figure(figsize=(10, 6))
                plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.title('Feature Importance')
                plt.gca().invert_yaxis()  # 降順に並べる
                st.pyplot(plt)
                plt.clf()

            with feature_importance_col2:
                corr_matrix = df_sample[model_feature_select].corr()

                # 相関係数の上位5つを取得するための関数
                def get_top_correlations(corr_matrix, top_n=5):
                    # 上三角行列をマスク
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    # 相関係数をマスクし、スタックしてSeriesに変換
                    corr_stack = corr_matrix.where(~mask).stack()
                    # 絶対値でソートし、上位5つを取得
                    top_correlations = corr_stack.abs().nlargest(top_n)
                    # 上位5つの相関係数をデータフレームに変換
                    df_top_corr = top_correlations.reset_index()
                    df_top_corr.columns = ['Feature1', 'Feature2', 'Correlation']
                    return df_top_corr

                # 上位5つの相関係数を取得
                st.write("相関係数TOP5")
                top_5_corr = get_top_correlations(corr_matrix, top_n=5)

                top_5_corr

            st.markdown(
                f"""
                モデルの学習から上記の特徴量重要度を得られました。<br>
                相関係数の高い組み合わせと合わせてどの特徴量を深堀するか検討し、前処理・特徴量エンジニアリングを行ってください。
                """, unsafe_allow_html=True)     
            
    else:
        pass
#-----------------------------------
else:
    pass