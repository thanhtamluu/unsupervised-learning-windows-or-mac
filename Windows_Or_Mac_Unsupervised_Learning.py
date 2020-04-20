{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas            as pd                          # data science essentials\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt                         # fundamental data visualization\n",
    "import seaborn           as sns                         # enhanced visualizations\n",
    "from sklearn.preprocessing import StandardScaler        # standard scaler\n",
    "from sklearn.decomposition import PCA                   # pca\n",
    "from scipy.cluster.hierarchy import dendrogram, linkage # dendrograms\n",
    "from sklearn.cluster         import KMeans              # k-means clustering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "########################################\n",
    "# loading data and setting display options\n",
    "########################################\n",
    "# loading data\n",
    "survey_df = pd.read_excel('./survey_data-1.xlsx')\n",
    "\n",
    "\n",
    "# setting print options\n",
    "pd.set_option('display.max_rows', 500)\n",
    "pd.set_option('display.max_columns', 500)\n",
    "pd.set_option('display.width', 1000)\n",
    "pd.set_option('display.max_colwidth', 100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df = survey_df.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "#process datasets\n",
    "acceptable_difference = 3\n",
    " \n",
    "def Grouping_features(df, cols, type = 'same'):\n",
    "    \n",
    "    for i in cols:\n",
    "        if type =='same':\n",
    "            df.loc[:, 'dif_'+ i[0]] = abs(df.loc[:, i[0]] - df.loc[:,i[1]])\n",
    "            df = df[df['dif_'+ i[0]] < acceptable_difference]\n",
    "           \n",
    "        elif type == 'similar':\n",
    "            df.loc[:, 'dif_'+i[0]] = abs(df.loc[:, i[0]] - df.loc[:,i[1]])\n",
    "            df = df[df['dif_'+ i[0]] < acceptable_difference +1]\n",
    "           \n",
    "        elif type == 'opposite':\n",
    "            df.loc[:, 'dif_'+i[0]] = abs(df.loc[:, i[0]] - abs(df.loc[:,i[1]]-6))\n",
    "            df = df[df['dif_'+ i[0]] < acceptable_difference]\n",
    "           \n",
    "        elif type == 'similar_opposite':\n",
    "            df.loc[:, 'dif_'+i[0]] = abs(df.loc[:, i[0]] + df.loc[:,i[1]]-abs(df.loc[:,i[2]]-6))\n",
    "            df = df[df['dif_'+ i[0]] < acceptable_difference*2]\n",
    "           \n",
    "        else:\n",
    "            return df\n",
    "           \n",
    "    df=df.drop(i, axis=1)\n",
    "    df=df.drop('dif_'+i[0], axis=1)\n",
    "       \n",
    "    print(df.shape)\n",
    "    \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['surveyID', 'Am the life of the party', 'Feel little concern for others', 'Am always prepared', 'Get stressed out easily', 'Have a rich vocabulary', 'Don't talk a lot', 'Am interested in people', 'Leave my belongings around', 'Am relaxed most of the time', 'Have difficulty understanding abstract ideas', 'Feel comfortable around people', 'Insult people', 'Pay attention to details', 'Worry about things', 'Have a vivid imagination', 'Keep in the background', 'Sympathize with others' feelings', 'Make a mess of things', 'Seldom feel blue', 'Am not interested in abstract ideas', 'Start conversations', 'Am not interested in other people's problems', 'Get chores done right away', 'Am easily disturbed', 'Have excellent ideas', 'Have little to say', 'Have a soft heart', 'Often forget to put things back in their proper place', 'Get upset easily', 'Do not have a good imagination', 'Talk to a lot of different people at parties', 'Am not really interested in others', 'Like order',\n",
       "       'Change my mood a lot', 'Am quick to understand things', 'Don't like to draw attention to myself', 'Take time out for others', 'Shirk my duties', 'Have frequent mood swings', 'Use difficult words', 'Don't mind being the center of attention', 'Feel others' emotions', 'Follow a schedule', 'Get irritated easily', 'Spend time reflecting on things', 'Am quiet around strangers', 'Make people feel at ease', 'Am exacting in my work', 'Often feel blue', 'Am full of ideas', 'See underlying patterns in complex situations', 'Don't  generate ideas that are new and different', 'Demonstrate an awareness of personal strengths and limitations', 'Display a growth mindset', 'Respond effectively to multiple priorities', 'Take initiative even when circumstances, objectives, or rules aren't clear', 'Encourage direct and open discussions', 'Respond effectively to multiple priorities.1', 'Take initiative even when circumstances, objectives, or rules aren't clear.1',\n",
       "       'Encourage direct and open discussions.1', 'Listen carefully to others', 'Don't persuasively sell a vision or idea', 'Build cooperative relationships', 'Work well with people from diverse cultural backgrounds', 'Effectively negotiate interests, resources, and roles', 'Can't rally people on the team around a common goal', 'Translate ideas into plans that are organized and realistic', 'Resolve conflicts constructively', 'Seek and use feedback from teammates', 'Coach teammates for performance and growth', 'Drive for results', 'What laptop do you currently have?', 'What laptop would you buy in next assuming if all laptops cost the same?', 'What program are you in?', 'What is your age?', 'Gender', 'What is your nationality? ', 'What is your ethnicity?'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "survey_df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# survey_df['Take initiative even when circumstances, objectives, or rules aren\\'t clear']\n",
    "same=[\n",
    "    ['Encourage direct and open discussions','Encourage direct and open discussions.1'],\n",
    "    ['Respond effectively to multiple priorities','Respond effectively to multiple priorities.1'],\n",
    "    ['Take initiative even when circumstances, objectives, or rules aren\\'t clear', 'Take initiative even when circumstances, objectives, or rules aren\\'t clear.1']\n",
    "]\n",
    " \n",
    "similar=[\n",
    "['Am not interested in other people\\'s problems','Feel little concern for others'],\n",
    "['Have frequent mood swings' , 'Change my mood a lot'],\n",
    "['Have a rich vocabulary', 'Use difficult words'],\n",
    "['Sympathize with others\\' feelings','Feel others\\' emotions'],\n",
    "['Don\\'t talk a lot', 'Have little to say'],\n",
    "['Often forget to put things back in their proper place', 'Leave my belongings around']\n",
    "]\n",
    " \n",
    "opposites=[['Have a vivid imagination','Do not have a good imagination'],\n",
    "['Seldom feel blue','Often feel blue '],\n",
    "['Don\\'t mind being the center of attention','Don\\'t like to draw attention to myself'],\n",
    "['Get chores done right away','Shirk my duties'],\n",
    "['Am interested in people','Am not really interested in others '],\n",
    "['Talk to a lot of different people at parties','Am quiet around strangers'],\n",
    "['Am the life of the party','Keep in the background'],\n",
    "['Like order','Make a mess of things'],\n",
    "['Have excellent ideas','Am full of ideas']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(146, 79)\n",
      "(138, 82)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/oukusunoki/opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:844: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  self.obj[key] = _infer_fill_value(value)\n",
      "/Users/oukusunoki/opt/anaconda3/lib/python3.7/site-packages/pandas/core/indexing.py:965: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  self.obj[item] = s\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(138, 82)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset = survey_df.copy()\n",
    "\n",
    "dataset = Grouping_features(dataset,same,type = 'same')\n",
    "dataset = Grouping_features(dataset,similar,type = 'similar')\n",
    "dataset = Grouping_features(dataset,opposites,type = 'opposites')\n",
    "dataset.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "########################################\n",
    "# inertia\n",
    "########################################\n",
    "def interia_plot(data, max_clust = 50):\n",
    "    \"\"\"\n",
    "PARAMETERS\n",
    "----------\n",
    "data      : DataFrame, data from which to build clusters. Dataset should be scaled\n",
    "max_clust : int, maximum of range for how many clusters to check interia, default 50\n",
    "    \"\"\"\n",
    "\n",
    "    ks = range(1, max_clust)\n",
    "    inertias = []\n",
    "\n",
    "\n",
    "    for k in ks:\n",
    "        # INSTANTIATING a kmeans object\n",
    "        model = KMeans(n_clusters = k)\n",
    "\n",
    "\n",
    "        # FITTING to the data\n",
    "        model.fit(data)\n",
    "\n",
    "\n",
    "        # append each inertia to the list of inertias\n",
    "        inertias.append(model.inertia_)\n",
    "\n",
    "\n",
    "\n",
    "    # plotting ks vs inertias\n",
    "    fig, ax = plt.subplots(figsize = (12, 8))\n",
    "    plt.plot(ks, inertias, '-o')\n",
    "\n",
    "\n",
    "    # labeling and displaying the plot\n",
    "    plt.xlabel('number of clusters, k')\n",
    "    plt.ylabel('inertia')\n",
    "    plt.xticks(ks)\n",
    "    plt.show()\n",
    "\n",
    " \n",
    "########################################\n",
    "# scree_plot\n",
    "########################################\n",
    "def scree_plot(pca_object, export = False):\n",
    "    # building a scree plot\n",
    " \n",
    "    # setting plot size\n",
    "    fig, ax = plt.subplots(figsize=(10, 8))\n",
    "    features = range(pca_object.n_components_)\n",
    " \n",
    " \n",
    "    # developing a scree plot\n",
    "    plt.plot(features,\n",
    "             pca_object.explained_variance_ratio_,\n",
    "             linewidth = 2,\n",
    "             marker = 'o',\n",
    "             markersize = 10,\n",
    "             markeredgecolor = 'black',\n",
    "             markerfacecolor = 'grey')\n",
    " \n",
    " \n",
    "    # setting more plot options\n",
    "    plt.title('Scree Plot')\n",
    "    plt.xlabel('PCA feature')\n",
    "    plt.ylabel('Explained Variance')\n",
    "    plt.xticks(features)\n",
    " \n",
    "    if export == True:\n",
    "   \n",
    "        # exporting the plot\n",
    "        plt.savefig('top_customers_correlation_scree_plot.png')\n",
    "       \n",
    "    # displaying the plot\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "p_5 = survey_df.loc[:,[\n",
    "    'surveyID', 'Am the life of the party', 'Feel little concern for others', 'Am always prepared', 'Get stressed out easily', 'Have a rich vocabulary', 'Don\\'t talk a lot', 'Am interested in people', 'Leave my belongings around', 'Am relaxed most of the time', 'Have difficulty understanding abstract ideas', 'Feel comfortable around people', 'Insult people', 'Pay attention to details', 'Worry about things', 'Have a vivid imagination', 'Keep in the background', 'Sympathize with others\\' feelings', 'Make a mess of things', 'Seldom feel blue', 'Am not interested in abstract ideas', 'Start conversations', 'Am not interested in other people\\'s problems', 'Get chores done right away', 'Am easily disturbed', 'Have excellent ideas', 'Have little to say', 'Have a soft heart', 'Often forget to put things back in their proper place', 'Get upset easily', 'Do not have a good imagination', 'Talk to a lot of different people at parties', 'Am not really interested in others', 'Like order',\n",
    "       'Change my mood a lot', 'Am quick to understand things', 'Don\\'t like to draw attention to myself', 'Take time out for others', 'Shirk my duties', 'Have frequent mood swings', 'Use difficult words', 'Don\\'t mind being the center of attention', 'Feel others\\' emotions', 'Follow a schedule', 'Get irritated easily', 'Spend time reflecting on things', 'Am quiet around strangers', 'Make people feel at ease', 'Am exacting in my work', 'Often feel blue', 'Am full of ideas'\n",
    "]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(147, 6)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>surveyID</th>\n",
       "      <th>E</th>\n",
       "      <th>A</th>\n",
       "      <th>C</th>\n",
       "      <th>N</th>\n",
       "      <th>O</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a1000</td>\n",
       "      <td>22</td>\n",
       "      <td>3</td>\n",
       "      <td>32</td>\n",
       "      <td>21</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>a1001</td>\n",
       "      <td>15</td>\n",
       "      <td>4</td>\n",
       "      <td>29</td>\n",
       "      <td>20</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>a1002</td>\n",
       "      <td>24</td>\n",
       "      <td>2</td>\n",
       "      <td>29</td>\n",
       "      <td>33</td>\n",
       "      <td>23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>a1003</td>\n",
       "      <td>18</td>\n",
       "      <td>2</td>\n",
       "      <td>22</td>\n",
       "      <td>20</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>a1004</td>\n",
       "      <td>21</td>\n",
       "      <td>1</td>\n",
       "      <td>22</td>\n",
       "      <td>17</td>\n",
       "      <td>23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>a1005</td>\n",
       "      <td>14</td>\n",
       "      <td>5</td>\n",
       "      <td>18</td>\n",
       "      <td>11</td>\n",
       "      <td>23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>a1006</td>\n",
       "      <td>18</td>\n",
       "      <td>4</td>\n",
       "      <td>22</td>\n",
       "      <td>13</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>a1007</td>\n",
       "      <td>20</td>\n",
       "      <td>4</td>\n",
       "      <td>11</td>\n",
       "      <td>28</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>a1008</td>\n",
       "      <td>29</td>\n",
       "      <td>4</td>\n",
       "      <td>30</td>\n",
       "      <td>31</td>\n",
       "      <td>29</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>a1009</td>\n",
       "      <td>17</td>\n",
       "      <td>6</td>\n",
       "      <td>29</td>\n",
       "      <td>23</td>\n",
       "      <td>21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>a1010</td>\n",
       "      <td>24</td>\n",
       "      <td>7</td>\n",
       "      <td>39</td>\n",
       "      <td>24</td>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>a1011</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "      <td>26</td>\n",
       "      <td>10</td>\n",
       "      <td>26</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>a1012</td>\n",
       "      <td>20</td>\n",
       "      <td>10</td>\n",
       "      <td>27</td>\n",
       "      <td>25</td>\n",
       "      <td>31</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>a1013</td>\n",
       "      <td>26</td>\n",
       "      <td>3</td>\n",
       "      <td>30</td>\n",
       "      <td>29</td>\n",
       "      <td>23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>a1014</td>\n",
       "      <td>13</td>\n",
       "      <td>5</td>\n",
       "      <td>27</td>\n",
       "      <td>16</td>\n",
       "      <td>32</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>a1015</td>\n",
       "      <td>15</td>\n",
       "      <td>5</td>\n",
       "      <td>27</td>\n",
       "      <td>18</td>\n",
       "      <td>26</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>a1016</td>\n",
       "      <td>22</td>\n",
       "      <td>4</td>\n",
       "      <td>28</td>\n",
       "      <td>26</td>\n",
       "      <td>28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>a1017</td>\n",
       "      <td>25</td>\n",
       "      <td>4</td>\n",
       "      <td>34</td>\n",
       "      <td>36</td>\n",
       "      <td>25</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>a1018</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "      <td>24</td>\n",
       "      <td>12</td>\n",
       "      <td>24</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>a1019</td>\n",
       "      <td>29</td>\n",
       "      <td>4</td>\n",
       "      <td>23</td>\n",
       "      <td>20</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   surveyID   E   A   C   N   O\n",
       "0     a1000  22   3  32  21  25\n",
       "1     a1001  15   4  29  20  30\n",
       "2     a1002  24   2  29  33  23\n",
       "3     a1003  18   2  22  20  29\n",
       "4     a1004  21   1  22  17  23\n",
       "5     a1005  14   5  18  11  23\n",
       "6     a1006  18   4  22  13  25\n",
       "7     a1007  20   4  11  28  15\n",
       "8     a1008  29   4  30  31  29\n",
       "9     a1009  17   6  29  23  21\n",
       "10    a1010  24   7  39  24  31\n",
       "11    a1011  22   4  26  10  26\n",
       "12    a1012  20  10  27  25  31\n",
       "13    a1013  26   3  30  29  23\n",
       "14    a1014  13   5  27  16  32\n",
       "15    a1015  15   5  27  18  26\n",
       "16    a1016  22   4  28  26  28\n",
       "17    a1017  25   4  34  36  25\n",
       "18    a1018  21   0  24  12  24\n",
       "19    a1019  29   4  23  20  20"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# \n",
    "\n",
    "big5_p = pd.DataFrame()\n",
    "big5_p['surveyID'] = p_5['surveyID']\n",
    "big5_p['E'] = 20 \\\n",
    "    + p_5.iloc[:,  1] - p_5.iloc[:, 6]\\\n",
    "    + p_5.iloc[:, 11] - p_5.iloc[:,16]\\\n",
    "    + p_5.iloc[:, 21] - p_5.iloc[:,26]\\\n",
    "    + p_5.iloc[:, 31] - p_5.iloc[:,36]\\\n",
    "    + p_5.iloc[:, 41] - p_5.iloc[:,46]\n",
    "\n",
    "big5_p['A'] = 14 \\\n",
    "    - p_5.iloc[:,  2] - p_5.iloc[:, 7]\\\n",
    "    - p_5.iloc[:, 12] - p_5.iloc[:,17]\\\n",
    "    - p_5.iloc[:, 22] - p_5.iloc[:,27]\\\n",
    "    - p_5.iloc[:, 32] + p_5.iloc[:,37]\\\n",
    "    + p_5.iloc[:, 42] + p_5.iloc[:,47]\n",
    "\n",
    "big5_p['C'] = 14 \\\n",
    "    + p_5.iloc[:,  3] - p_5.iloc[:, 8]\\\n",
    "    + p_5.iloc[:, 13] - p_5.iloc[:,18]\\\n",
    "    + p_5.iloc[:, 23] - p_5.iloc[:,28]\\\n",
    "    + p_5.iloc[:, 33] - p_5.iloc[:,38]\\\n",
    "    + p_5.iloc[:, 43] + p_5.iloc[:,48]\n",
    "    \n",
    "big5_p['N'] = 38 \\\n",
    "    - p_5.iloc[:,  4] + p_5.iloc[:, 9]\\\n",
    "    - p_5.iloc[:, 14] + p_5.iloc[:,19]\\\n",
    "    - p_5.iloc[:, 24] - p_5.iloc[:,29]\\\n",
    "    - p_5.iloc[:, 34] - p_5.iloc[:,39]\\\n",
    "    - p_5.iloc[:, 44] - p_5.iloc[:,49]\n",
    "    \n",
    "big5_p['O'] = 8 \\\n",
    "    + p_5.iloc[:,  5] - p_5.iloc[:,10]\\\n",
    "    + p_5.iloc[:, 15] - p_5.iloc[:,20]\\\n",
    "    + p_5.iloc[:, 25] - p_5.iloc[:,30]\\\n",
    "    + p_5.iloc[:, 35] + p_5.iloc[:,40]\\\n",
    "    + p_5.iloc[:, 45] + p_5.iloc[:,50]\n",
    "\n",
    "\n",
    "print(big5_p.shape)\n",
    "\n",
    "big5_p.head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(138, 6)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# drop unrelaiable\n",
    "# big5_p = big5_p[big5_p['surveyID'] in dataset['surveyID']]\n",
    "\n",
    "\n",
    "big5_p = big5_p.loc[big5_p['surveyID'].isin(dataset['surveyID'])]\n",
    "big5_p.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "big_5_X\n",
      " E    39.909263\n",
      "A    10.640832\n",
      "C    30.926118\n",
      "N    47.713085\n",
      "O    27.742544\n",
      "dtype: float64 \n",
      "\n",
      "\n",
      "big_5_X_scaler\n",
      " E    1.0\n",
      "A    1.0\n",
      "C    1.0\n",
      "N    1.0\n",
      "O    1.0\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "big_5_X = big5_p.drop('surveyID',axis = 1)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_scaler = scaler.fit_transform(big_5_X)\n",
    " \n",
    "big_5_X_scaler = pd.DataFrame(X_scaler)\n",
    "big_5_X_scaler.columns = big_5_X.columns\n",
    "\n",
    "\n",
    "print('big_5_X\\n',np.var(big_5_X),'\\n\\n')\n",
    "print('big_5_X_scaler\\n',np.var(big_5_X_scaler))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "big5_pca = PCA(n_components=None, random_state=802)\n",
    "big5_pca_fit = big5_pca.fit_transform(big_5_X_scaler)\n",
    "\n",
    "scree_plot(pca_object = big5_pca)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAmcAAAHwCAYAAADjOch3AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nOzdd3iV9f3/8ef7nCzCCiPsvcQgO7IStGqtWpkOBMWtgJJQa23Var9WWzu0tZahiEodIIgoK65ata1hh72RvfdeCUk+vz9y6C9qCAFycp+TvB7XdS7OvU5eeCm+uM99v29zziEiIiIiocHndQARERER+f9UzkRERERCiMqZiIiISAhRORMREREJISpnIiIiIiFE5UxEREQkhKiciYgEiZk1MjNnZhFeZxGR8KFyJiJhw8ySzWyWmR02swNmNtPMLvc404/MLNfMjpnZUTNbY2b3XsDn/NbMxgUjo4iEF/1tTkTCgplVAtKAh4BJQBTQHcg8z8+JcM5lF3O8Hc65emZmQG9gspnNBU4U888RkTJAZ85EJFy0AHDOTXDO5TjnTjrn/umcW3pmBzN70MxWBc5grTSzDoH1m8zscTNbChw3swgzq2NmH5rZXjPbaGbD8n2Oz8yeMLP1ZrbfzCaZWdVzBXR5pgIHgYTvbw/8zOmBs37rzOzBwPrrgV8DtwXOwC25yH9WIhLGVM5EJFysBXLM7G0zu8HMquTfaGa3Ar8F7gIqAb2A/fl2GQDcCMQBucAMYAlQF7gGeMTMrgvsOwzoA1wJ1CGvbI06V8BAqesb+BnLCthlArAt8Jm3AH8ws2ucc58BfwDed85VcM61PdfPEpHSS+VMRMKCc+4IkAw44HVgb+AsVM3ALg8ALzjn5gfOYK1zzm3O9xHDnXNbnXMngcuBeOfcc865LOfchsBn9g/sOxh4yjm3zTmXSV7pu6WQC/vrmNkhYB/wDHCnc25N/h3MrH4g/+POuVPOucXAG8CdF/PPRURKH11zJiJhwzm3CrgHwMxaAuOAl8k7K1YfWF/I4VvzvW/I/y9UZ/iBb/Jtn2Jmufm25wA1ge0FfPYO51y9c8SvAxxwzh3Nt24zkHiO40SkjFE5E5Gw5JxbbWZvkXeWC/LKV9PCDsn3fiuw0TnX/Cz7bgXuc87NvOig/98OoKqZVcxX0Brw/8ueK/gwESlr9LWmiIQFM2tpZr8ws3qB5frknTGbE9jlDeAxM+toeZqZWcOzfNw84EjgJoFyZuY3s8vyjeUYDTx/5ngzizez3heT3zm3FZgF/NHMYsysDXA/MD6wy26gkZnpz2WRMk5/CIhIuDgKdAbmmtlx8krZcuAXAM65D4DngfcC+04FCrzD0jmXA/QE2gEbybtW7A2gcmCXvwPTgX+a2dHAz+pcDL+HAUAj8s6iTQGecc59Edj2QeDX/Wa2sBh+loiEKXNOZ9JFREREQoXOnImIiIiEEJUzERERkRCiciYiIiISQlTOREREREKIypmIiIhICCk1Q2irV6/uGjVq5HUMERERkXNasGDBPudcfEHbSk05a9SoERkZGV7HEBERETknM9t8tm36WlNEREQkhKiciYiIiIQQlTMRERGREKJyJiIiIhJCVM5EREREQojKmYiIiEgIUTkTERERCSEqZyIiIiIhROVMREREJISonImIiIiEEJUzERERkRCiciYiIiISQlTOREREREKIylkRrF+/ntTUVKpVq4bf76datWqkpqayfv16r6OJiIhIKRPhdYBQ9+mnnzJgwADatWvHwIEDqVy5MocPH2bJkiV07NiRCRMmcMMNN3gdU0REREoJlbNCrF+/ngEDBnDLLbdQv379/62vWrUqV111Fc2aNWPAgAEsWLCApk2bephURERESgt9rVmIl19+mXbt2n2nmOVXv3592rZty/Dhw0s4mYiIiJRWKmeFeO+992jbtm2h+7Rr147x48eXUCIREREp7VTOCnHo0CEqV65c6D6VK1fm0KFDJZRIRERESjuVs0LExcVx+PDhQvc5fPgwcXFxJZRIRERESjuVs0LcfvvtLFmypNB9MhYu4vo+t5RQIhERESntVM4K8cgjj7B48WK2bt1a4PatW7eSsXARM6M7MXnBthJOJyIiIqWRylkhmjZtyoQJE5g8eTJfffUVBw4cICcnhwMHDvDVV18xefJkej3yZ1zFmjz2wRKemrKMzOwcr2OLiIhIGNOcs3O44YYbWLBgAcOHD2f8+PEcOnSIuLg47rjjDsaOHUvTpk15f/4WfjNtBePnbmH5jiO8ckcH6saV8zq6iIiIhCFzznmdoVgkJia6jIwMz37+sm2HGTJuAdsPnaRq+SiG929PcvPqnuURERGR0GVmC5xziQVt09eaxaR1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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "big5_pca_4 = PCA(n_components=2, random_state=802)\n",
    "big5_pca_fit_4 = big5_pca_4.fit_transform(big_5_X_scaler)\n",
    "\n",
    "scree_plot(pca_object = big5_pca_4)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/oukusunoki/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead\n",
      "  \"\"\"\n"
     ]
    }
   ],
   "source": [
    "\n",
    "####################\n",
    "### Max PC Model ###\n",
    "####################\n",
    "# transposing pca components (pc = MAX)\n",
    "factor_loadings = pd.DataFrame(pd.np.transpose(big5_pca\n",
    "                                               \n",
    "                                               \n",
    "                                               \n",
    "                                               .components_))\n",
    "\n",
    "\n",
    "# naming rows as original features\n",
    "factor_loadings = factor_loadings.set_index(big_5_X_scaler.columns)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "MAX Components Factor Loadings\n",
      "------------------------------\n",
      "      0     1     2     3     4\n",
      "E -0.26 -0.84 -0.12 -0.41 -0.22\n",
      "A -0.53  0.30  0.08 -0.58  0.53\n",
      "C -0.59  0.33  0.11  0.06 -0.73\n",
      "N -0.33  0.02 -0.87  0.33  0.16\n",
      "O -0.45 -0.31  0.45  0.62  0.34\n",
      "\n",
      "\n",
      "2 Components Factor Loadings\n",
      "------------------------------\n",
      "      0     1\n",
      "E -0.26 -0.84\n",
      "A -0.53  0.30\n",
      "C -0.59  0.33\n",
      "N -0.33  0.02\n",
      "O -0.45 -0.31\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/oukusunoki/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:5: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead\n",
      "  \"\"\"\n"
     ]
    }
   ],
   "source": [
    "##################\n",
    "### 3 PC Model ###\n",
    "##################\n",
    "# transposing pca components (pc = 3)\n",
    "factor_loadings_custom = pd.DataFrame(pd.np.transpose(big5_pca_4.components_))\n",
    "\n",
    "\n",
    "# naming rows as original features\n",
    "factor_loadings_custom = factor_loadings_custom.set_index(big_5_X_scaler.columns)\n",
    "\n",
    "\n",
    "# checking the results\n",
    "print(f\"\"\"\n",
    "MAX Components Factor Loadings\n",
    "------------------------------\n",
    "{factor_loadings.round(2)}\n",
    "\n",
    "\n",
    "2 Components Factor Loadings\n",
    "------------------------------\n",
    "{factor_loadings_custom.round(2)}\n",
    "\"\"\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "PCA_components = pd.DataFrame(big5_pca_fit_4)\n",
    "\n",
    "PCA_components.columns = ['V_' + str(i) for i in range(PCA_components.shape[1])]\n",
    "\n",
    "# import plotly.express as px \n",
    "# df = px.data.iris()\n",
    "# fig = px.scatter_3d(PCA_components, x = 'V1', y = 'V2', z = 'V3')\n",
    "# fig.show()  \n",
    "\n",
    "# PCA_components.set_index(big_5.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "V_0    1.831624\n",
      "V_1    1.015689\n",
      "dtype: float64 \n",
      "\n",
      "\n",
      "V_0    1.0\n",
      "V_1    1.0\n",
      "dtype: float64\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/oukusunoki/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:18: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead\n",
      "/Users/oukusunoki/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:19: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# INSTANTIATING a StandardScaler() object\n",
    "scaler = StandardScaler()\n",
    "\n",
    "\n",
    "# FITTING and TRANSFORMING the scaled data\n",
    "X_scaled_pca = scaler.fit_transform(PCA_components)\n",
    "\n",
    "\n",
    "# converting scaled data into a DataFrame\n",
    "pca_scaled = pd.DataFrame(X_scaled_pca)\n",
    "\n",
    "\n",
    "# reattaching column names\n",
    "pca_scaled.columns = ['V_' + str(i) for i in range(PCA_components.shape[1])]\n",
    "\n",
    "\n",
    "# checking pre- and post-scaling variance\n",
    "print(pd.np.var(PCA_components), '\\n\\n')\n",
    "print(pd.np.var(pca_scaled))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x864 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "# grouping data based on Ward distance \n",
    "standard_mergings_ward = linkage(y = pca_scaled, \n",
    "                                 method = 'ward')\n",
    "\n",
    "# setting plot size \n",
    "fig, ax = plt.subplots(figsize = (12, 12))\n",
    "\n",
    "# developing a dendogram \n",
    "dendrogram(Z = standard_mergings_ward)\n",
    "\n",
    "# saving and displaying the plot \n",
    "plt.savefig('standard_hierarchical_clust_ward.png')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "interia_plot(data = pca_scaled, \n",
    "             max_clust = 20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2    53\n",
      "1    44\n",
      "0    41\n",
      "Name: Big 5 Clusters, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Big 5 Clusters</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Big 5 Clusters\n",
       "0               2\n",
       "1               2\n",
       "2               1\n",
       "3               0\n",
       "4               0"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# INSTANTIATING a k-Means object with four clusters\n",
    "customers_k_pca = KMeans(n_clusters = 3,\n",
    "                         random_state = 802)\n",
    "\n",
    "\n",
    "# fitting the object to the data\n",
    "customers_k_pca.fit(pca_scaled)\n",
    "\n",
    "\n",
    "# converting the clusters to a DataFrame\n",
    "customers_kmeans_pca = pd.DataFrame({'Big 5 Clusters': customers_k_pca.labels_})\n",
    "\n",
    "\n",
    "# checking the results\n",
    "print(customers_kmeans_pca.iloc[: , 0].value_counts())\n",
    "customers_kmeans_pca.head(n = 5)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.16370912, -0.34421715],\n",
       "       [-0.79368898, -0.82083169],\n",
       "       [-0.2413162 ,  0.94772637]])"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "centroids_pca = customers_k_pca.cluster_centers_\n",
    "centroids_pca"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>V_0</th>\n",
       "      <th>V_1</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1.16</td>\n",
       "      <td>-0.34</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-0.79</td>\n",
       "      <td>-0.82</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-0.24</td>\n",
       "      <td>0.95</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    V_0   V_1\n",
       "0  1.16 -0.34\n",
       "1 -0.79 -0.82\n",
       "2 -0.24  0.95"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# storing cluster centers \n",
    "centroids_pca = customers_k_pca.cluster_centers_\n",
    "\n",
    "# converting cluster centers into a DataFrame \n",
    "centroids_pca_df = pd.DataFrame(centroids_pca)\n",
    "\n",
    "# renaming principal components \n",
    "centroids_pca_df.columns = ['V_' + str(i) for i in range(centroids_pca_df.shape[1])]\n",
    "\n",
    "# checking results (clusters = rows, pc = columns)\n",
    "centroids_pca_df.round(2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>surveyID</th>\n",
       "      <th>What laptop do you currently have?</th>\n",
       "      <th>What laptop would you buy in next assuming if all laptops cost the same?</th>\n",
       "      <th>What program are you in?</th>\n",
       "      <th>What is your age?</th>\n",
       "      <th>Gender</th>\n",
       "      <th>What is your nationality?</th>\n",
       "      <th>What is your ethnicity?</th>\n",
       "      <th>Big 5 Clusters</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a1000</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>25</td>\n",
       "      <td>Female</td>\n",
       "      <td>ecuador</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>a1001</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>Ecuador</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>a1002</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>25</td>\n",
       "      <td>Male</td>\n",
       "      <td>Indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>a1003</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>26</td>\n",
       "      <td>Female</td>\n",
       "      <td>indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>a1004</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>DD (MBA &amp; Disruptive innovation)</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>China</td>\n",
       "      <td>Far east Asian</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>a1005</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>22</td>\n",
       "      <td>Male</td>\n",
       "      <td>Indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>a1006</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>23</td>\n",
       "      <td>Female</td>\n",
       "      <td>Dominican</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>a1007</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>23</td>\n",
       "      <td>Male</td>\n",
       "      <td>Belgian</td>\n",
       "      <td>White / Caucasian</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>a1008</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>25</td>\n",
       "      <td>Female</td>\n",
       "      <td>Swiss</td>\n",
       "      <td>White / Caucasian</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>a1009</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MBA &amp; Business Analytics)</td>\n",
       "      <td>38</td>\n",
       "      <td>Male</td>\n",
       "      <td>Japan</td>\n",
       "      <td>Far east Asian</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  surveyID What laptop do you currently have? What laptop would you buy in next assuming if all laptops cost the same?          What program are you in?  What is your age?  Gender What is your nationality?  What is your ethnicity?  Big 5 Clusters\n",
       "0    a1000                            Macbook                                                                  Macbook     DD (MIB & Business Analytics)                 25  Female                    ecuador       Hispanic / Latino             2.0\n",
       "1    a1001                     Windows laptop                                                           Windows laptop       One year Business Analytics                 27    Male                    Ecuador       Hispanic / Latino             2.0\n",
       "2    a1002                     Windows laptop                                                           Windows laptop       One year Business Analytics                 25    Male                     Indian     West Asian / Indian             1.0\n",
       "3    a1003                     Windows laptop                                                           Windows laptop       One year Business Analytics                 26  Female                     indian     West Asian / Indian             0.0\n",
       "4    a1004                            Macbook                                                           Windows laptop  DD (MBA & Disruptive innovation)                 27    Male                      China          Far east Asian             0.0\n",
       "5    a1005                            Macbook                                                                  Macbook     DD (MIB & Business Analytics)                 22    Male                     Indian     West Asian / Indian             0.0\n",
       "6    a1006                     Windows laptop                                                                  Macbook     DD (MIB & Business Analytics)                 23  Female                 Dominican        Hispanic / Latino             0.0\n",
       "7    a1007                            Macbook                                                                  Macbook     DD (MIB & Business Analytics)                 23    Male                    Belgian       White / Caucasian             0.0\n",
       "8    a1008                     Windows laptop                                                           Windows laptop     DD (MIB & Business Analytics)                 25  Female                      Swiss       White / Caucasian             1.0\n",
       "9    a1009                            Macbook                                                                  Macbook     DD (MBA & Business Analytics)                 38    Male                      Japan          Far east Asian             2.0"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# adding labels\n",
    "clst_pca_df = pd.concat([survey_df.loc[:, ['surveyID', 'What laptop do you currently have?', \n",
    "                                           'What laptop would you buy in next assuming if all laptops cost the same?',\n",
    "                                           'What program are you in?', \n",
    "                                           'What is your age?',\n",
    "                                           'Gender', \n",
    "                                           'What is your nationality? ',\n",
    "                                           'What is your ethnicity?']],\n",
    "                        customers_kmeans_pca],\n",
    "                        axis = 1)\n",
    "\n",
    "clst_pca_df.head(n = 10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#######\n",
    "#   HULT DNA LIST   #\n",
    " \n",
    "adaptive_thinking =['See underlying patterns in complex situations',\n",
    "                   'Don\\'t  generate ideas that are new and different',\n",
    "                   'Demonstrate an awareness of personal strengths and limitations',\n",
    "                   'Display a growth mindset',\n",
    "                   'Take initiative even when circumstances, objectives, or rules aren\\'t clear'\n",
    "                   ]\n",
    " \n",
    "relationship_building     =['Build cooperative relationships',\n",
    "                   'Work well with people from diverse cultural backgrounds',\n",
    "                   'Effectively negotiate interests, resources, and roles'\n",
    "                  ]\n",
    " \n",
    "communication     =['Encourage direct and open discussions.1',\n",
    "                   'Listen carefully to others',\n",
    "                   'Don\\'t persuasively sell a vision or idea'\n",
    "                  ]\n",
    " \n",
    "teamwork        =['Can\\'t rally people on the team around a common goal',\n",
    "                  'Seek and use feedback from teammates',\n",
    "                  'Coach teammates for performance and growth',\n",
    "                  'Resolve conflicts constructively'\n",
    "                ]\n",
    " \n",
    "execution       =['Translate ideas into plans that are organized and realistic',\n",
    "                  'Drive for results',\n",
    "                  'Respond effectively to multiple priorities'\n",
    "                 ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(147, 5)"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "hult_dna = pd.DataFrame()\n",
    "\n",
    "hult_df = survey_df.copy()\n",
    "\n",
    "hult_df['Don\\'t  generate ideas that are new and different'] = 6 - survey_df['Don\\'t  generate ideas that are new and different']\n",
    "\n",
    "hult_df['Don\\'t persuasively sell a vision or idea'] = 6 - survey_df['Don\\'t persuasively sell a vision or idea']\n",
    "\n",
    "for i, j in hult_df.iterrows():\n",
    "    adaptive_thinking\n",
    "    hult_dna.loc[i, 'adaptive_thinking'] = survey_df.loc[i, adaptive_thinking].mean()\n",
    "    hult_dna.loc[i, 'communication'] = survey_df.loc[i, communication].mean()\n",
    "    hult_dna.loc[i, 'relationship_building'] = survey_df.loc[i, relationship_building].mean()\n",
    "    hult_dna.loc[i, 'teamwork'] = survey_df.loc[i, teamwork].mean()\n",
    "    hult_dna.loc[i, 'execution'] = survey_df.loc[i, execution].mean()\n",
    "    \n",
    "hult_dna.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>surveyID</th>\n",
       "      <th>adaptive_thinking</th>\n",
       "      <th>communication</th>\n",
       "      <th>relationship_building</th>\n",
       "      <th>teamwork</th>\n",
       "      <th>execution</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a1000</td>\n",
       "      <td>3.6</td>\n",
       "      <td>3.666667</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.50</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>a1001</td>\n",
       "      <td>3.2</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.333333</td>\n",
       "      <td>4.25</td>\n",
       "      <td>4.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>a1002</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2.666667</td>\n",
       "      <td>3.666667</td>\n",
       "      <td>2.50</td>\n",
       "      <td>4.666667</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>a1003</td>\n",
       "      <td>3.8</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>5.00</td>\n",
       "      <td>4.333333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>a1004</td>\n",
       "      <td>3.4</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.333333</td>\n",
       "      <td>3.25</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  surveyID  adaptive_thinking  communication  relationship_building  teamwork  execution\n",
       "0    a1000                3.6       3.666667               4.000000      3.50   4.000000\n",
       "1    a1001                3.2       4.000000               3.333333      4.25   4.333333\n",
       "2    a1002                3.0       2.666667               3.666667      2.50   4.666667\n",
       "3    a1003                3.8       4.000000               5.000000      5.00   4.333333\n",
       "4    a1004                3.4       4.000000               3.333333      3.25   3.000000"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hult_dna = pd.concat([hult_df['surveyID'], \n",
    "                      hult_dna], \n",
    "                     axis = 1)\n",
    "\n",
    "hult_dna.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(138, 6)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "hult_dna = hult_dna.loc[hult_dna['surveyID'].isin(dataset['surveyID'])]\n",
    "hult_dna.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "V_0    1.831624\n",
      "V_1    1.015689\n",
      "dtype: float64 \n",
      "\n",
      "\n",
      "V_0    1.0\n",
      "V_1    1.0\n",
      "dtype: float64\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/oukusunoki/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:21: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead\n",
      "/Users/oukusunoki/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:22: FutureWarning: The pandas.np module is deprecated and will be removed from pandas in a future version. Import numpy directly instead\n"
     ]
    }
   ],
   "source": [
    "\n",
    "hult_dna = hult_dna.drop('surveyID', \n",
    "                        axis = 1)\n",
    "\n",
    "# INSTANTIATING a StandardScaler() object\n",
    "scaler = StandardScaler()\n",
    "\n",
    "\n",
    "# FITTING and TRANSFORMING the scaled data\n",
    "X_scaled = scaler.fit_transform(hult_dna)\n",
    "\n",
    "\n",
    "# converting scaled data into a DataFrame\n",
    "hult_dna_scaled = pd.DataFrame(X_scaled)\n",
    "\n",
    "\n",
    "# reattaching column names\n",
    "hult_dna_scaled.columns = hult_dna.columns\n",
    "\n",
    "\n",
    "# checking pre- and post-scaling variance\n",
    "print(pd.np.var(PCA_components), '\\n\\n')\n",
    "print(pd.np.var(pca_scaled))\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# page 4 middle\n",
    "# instatiating the PCA object with no limit to proncipal componennts\n",
    "hult_dna_pca= PCA(n_components = None, random_state = 802)\n",
    " \n",
    "#fitting and transforming the scaled data\n",
    "hult_dna_pca_0 = hult_dna_pca.fit_transform(hult_dna_scaled)\n",
    " \n",
    "# calling the scree plot function\n",
    "scree_plot(pca_object = hult_dna_pca)\n",
    " \n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "#instantiating the PCA object with no limit to proncipal components\n",
    "hult_dna_pca_custom = PCA(n_components = 2, random_state = 802)\n",
    " \n",
    "# fitting and transforming the purchases scaled\n",
    "hult_dna_pca_custom_0 = hult_dna_pca_custom.fit_transform(hult_dna_scaled)\n",
    " \n",
    "# calling the scree plot function\n",
    "scree_plot(pca_object = hult_dna_pca_custom)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAX Components Factor Loadings)\n",
      "__________________________________________\n",
      "      0     1     2     3     4\n",
      "0 -0.47  0.18 -0.41  0.72 -0.25\n",
      "1 -0.37 -0.80 -0.26 -0.06  0.38\n",
      "2 -0.49  0.10 -0.20 -0.65 -0.53\n",
      "3 -0.43 -0.18  0.85  0.19 -0.18\n",
      "4 -0.46  0.53  0.06 -0.16  0.69\n",
      " \n",
      "2 Components Factor Loadings\n",
      "____________________________________________\n",
      " \n",
      "                          0     1\n",
      "adaptive_thinking     -0.26 -0.84\n",
      "communication         -0.53  0.30\n",
      "relationship_building -0.59  0.33\n",
      "teamwork              -0.33  0.02\n",
      "execution             -0.45 -0.31\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "###################################################################\n",
    "###### MAC PC MODEL #########################################\n",
    "#################################################################\n",
    " \n",
    "#transposing PCA components(pc=MAX)\n",
    "factor_loadings = pd.DataFrame(np.transpose(hult_dna_pca.components_))\n",
    " \n",
    "# naming rows as original features\n",
    "factor_loading = factor_loadings.set_index(hult_dna_scaled.columns)\n",
    "\n",
    "\n",
    "\n",
    "############################################################\n",
    "############ 3 PC Model ################################\n",
    "#############################################################\n",
    "# transposing pca components (pc=3)\n",
    "\n",
    "factor_loading_custom = pd.DataFrame(np.transpose(hult_dna_pca_custom.components_))\n",
    " \n",
    "# naming rows as original features\n",
    "factor_loadings_custom= factor_loadings_custom.set_index(hult_dna_scaled.columns)\n",
    "# checking the results\n",
    "print(f\"\"\"MAX Components Factor Loadings)\n",
    "__________________________________________\n",
    "{factor_loadings.round(2)}\n",
    " \n",
    "2 Components Factor Loadings\n",
    "____________________________________________\n",
    " \n",
    "{factor_loadings_custom.round(2)}\n",
    "\"\"\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "#page 5\n",
    " \n",
    "Hult_DNA_PCA_components = pd.DataFrame(hult_dna_pca_custom_0)\n",
    "Hult_DNA_PCA_components.columns = ['hult_dna_' + str(i) for i in range(Hult_DNA_PCA_components.shape[1])]\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hult_dna_0    2.787783\n",
      "hult_dna_1    0.799479\n",
      "dtype: float64 \n",
      "\n",
      "\n",
      "1.7936306999421676\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# import plotly expres as px\n",
    "# df= px.data.iris()\n",
    "#fig = px.scatter_3d(PCA_components, x='V1', y='V2', z='V3')\n",
    "# fig.show()\n",
    "# PCA_components.set_index(big_5.columns)\n",
    " \n",
    "# instatntiating the standard scaler model\n",
    "scaler = StandardScaler()\n",
    "X_scaleed_pca = scaler.fit_transform(Hult_DNA_PCA_components)\n",
    "pca_scaled = pd.DataFrame(X_scaled_pca)\n",
    "# pca_scaled.columns=['V_'+str(i) for i in range(hult_dna_pca_custom_0.shape[1])]\n",
    "\n",
    "\n",
    "print(np.var(Hult_DNA_PCA_components), '\\n\\n')\n",
    "print(np.var(hult_dna_pca_custom_0))\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x864 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# grouping databased on Ward distance\n",
    "standard_mergings_ward = linkage(y=hult_dna_pca_custom_0,method='ward')\n",
    " \n",
    "# setting plot soze\n",
    "fig, ax = plt.subplots(figsize = (12, 12))\n",
    "# DEVELOPING A DENDROGRAM\n",
    " \n",
    "dendrogram(Z=standard_mergings_ward, leaf_rotation = 90, leaf_font_size = 6)\n",
    "#saving and displaying the plot\n",
    "plt.savefig('standard_hierarchical_clust_ward.png')\n",
    "plt.show()\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "#calling the interia plot function\n",
    "interia_plot(data=hult_dna_pca_custom_0, max_clust=20)\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2    51\n",
      "3    35\n",
      "0    31\n",
      "1    21\n",
      "Name: Hult_DNA_Cluster, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hult_DNA_Cluster</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hult_DNA_Cluster\n",
       "0                 2\n",
       "1                 2\n",
       "2                 1\n",
       "3                 0\n",
       "4                 2"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# instatntiating a k-Means object with five clusters\n",
    "customers_k_pca =KMeans(n_clusters=4, random_state=802)\n",
    " \n",
    "#fitting the object to the data\n",
    "customers_k_pca.fit(hult_dna_pca_custom_0)\n",
    "#converting the clusters to a DataFrame\n",
    "customers_kmeans_pca = pd.DataFrame({'Hult_DNA_Cluster': customers_k_pca.labels_})\n",
    "# checking the results\n",
    "print(customers_kmeans_pca.iloc[:, 0].value_counts())\n",
    "customers_kmeans_pca.head(5)\n",
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>surveyID</th>\n",
       "      <th>What laptop do you currently have?</th>\n",
       "      <th>What laptop would you buy in next assuming if all laptops cost the same?</th>\n",
       "      <th>What program are you in?</th>\n",
       "      <th>What is your age?</th>\n",
       "      <th>Gender</th>\n",
       "      <th>What is your nationality?</th>\n",
       "      <th>What is your ethnicity?</th>\n",
       "      <th>Big 5 Clusters</th>\n",
       "      <th>Hult_DNA_Cluster</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a1000</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>25</td>\n",
       "      <td>Female</td>\n",
       "      <td>ecuador</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>a1001</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>Ecuador</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>a1002</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>25</td>\n",
       "      <td>Male</td>\n",
       "      <td>Indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>a1003</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>26</td>\n",
       "      <td>Female</td>\n",
       "      <td>indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>a1004</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>DD (MBA &amp; Disruptive innovation)</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>China</td>\n",
       "      <td>Far east Asian</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>a1005</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>22</td>\n",
       "      <td>Male</td>\n",
       "      <td>Indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>a1006</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>23</td>\n",
       "      <td>Female</td>\n",
       "      <td>Dominican</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>a1007</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>23</td>\n",
       "      <td>Male</td>\n",
       "      <td>Belgian</td>\n",
       "      <td>White / Caucasian</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>a1008</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>25</td>\n",
       "      <td>Female</td>\n",
       "      <td>Swiss</td>\n",
       "      <td>White / Caucasian</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>a1009</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MBA &amp; Business Analytics)</td>\n",
       "      <td>38</td>\n",
       "      <td>Male</td>\n",
       "      <td>Japan</td>\n",
       "      <td>Far east Asian</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  surveyID What laptop do you currently have? What laptop would you buy in next assuming if all laptops cost the same?          What program are you in?  What is your age?  Gender What is your nationality?  What is your ethnicity?  Big 5 Clusters  Hult_DNA_Cluster\n",
       "0    a1000                            Macbook                                                                  Macbook     DD (MIB & Business Analytics)                 25  Female                    ecuador       Hispanic / Latino             2.0               2.0\n",
       "1    a1001                     Windows laptop                                                           Windows laptop       One year Business Analytics                 27    Male                    Ecuador       Hispanic / Latino             2.0               2.0\n",
       "2    a1002                     Windows laptop                                                           Windows laptop       One year Business Analytics                 25    Male                     Indian     West Asian / Indian             1.0               1.0\n",
       "3    a1003                     Windows laptop                                                           Windows laptop       One year Business Analytics                 26  Female                     indian     West Asian / Indian             0.0               0.0\n",
       "4    a1004                            Macbook                                                           Windows laptop  DD (MBA & Disruptive innovation)                 27    Male                      China          Far east Asian             0.0               2.0\n",
       "5    a1005                            Macbook                                                                  Macbook     DD (MIB & Business Analytics)                 22    Male                     Indian     West Asian / Indian             0.0               2.0\n",
       "6    a1006                     Windows laptop                                                                  Macbook     DD (MIB & Business Analytics)                 23  Female                 Dominican        Hispanic / Latino             0.0               3.0\n",
       "7    a1007                            Macbook                                                                  Macbook     DD (MIB & Business Analytics)                 23    Male                    Belgian       White / Caucasian             0.0               0.0\n",
       "8    a1008                     Windows laptop                                                           Windows laptop     DD (MIB & Business Analytics)                 25  Female                      Swiss       White / Caucasian             1.0               2.0\n",
       "9    a1009                            Macbook                                                                  Macbook     DD (MBA & Business Analytics)                 38    Male                      Japan          Far east Asian             2.0               2.0"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# storing cluster centers\n",
    "centroinds_pca = customers_k_pca.cluster_centers_\n",
    " \n",
    "# converting cluster centers intyto a data frame\n",
    "centroind_pca_df = pd.DataFrame(centroinds_pca)\n",
    "#renaming principal components\n",
    "centroind_pca_df.columns = ['hult_'+str(i) for i in range(centroind_pca_df.shape[1])]\n",
    " \n",
    "#checking results (clusters = rows, pc=columns)\n",
    "centroids_pca_df.round(2)\n",
    " \n",
    "# add label\n",
    "clst_pca_df = pd.concat([clst_pca_df, customers_kmeans_pca], axis =1)\n",
    "clst_pca_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>surveyID</th>\n",
       "      <th>What laptop do you currently have?</th>\n",
       "      <th>What laptop would you buy in next assuming if all laptops cost the same?</th>\n",
       "      <th>What program are you in?</th>\n",
       "      <th>What is your age?</th>\n",
       "      <th>Gender</th>\n",
       "      <th>What is your nationality?</th>\n",
       "      <th>What is your ethnicity?</th>\n",
       "      <th>Big 5 Clusters</th>\n",
       "      <th>Hult_DNA_Cluster</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a1000</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>25</td>\n",
       "      <td>Female</td>\n",
       "      <td>ecuador</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>Mobile</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>a1001</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>Ecuador</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>Mobile</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>a1002</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>25</td>\n",
       "      <td>Male</td>\n",
       "      <td>Indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>Online</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>a1003</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>26</td>\n",
       "      <td>Female</td>\n",
       "      <td>indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>a1004</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>DD (MBA &amp; Disruptive innovation)</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>China</td>\n",
       "      <td>Far east Asian</td>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  surveyID What laptop do you currently have? What laptop would you buy in next assuming if all laptops cost the same?          What program are you in?  What is your age?  Gender What is your nationality?  What is your ethnicity? Big 5 Clusters  Hult_DNA_Cluster\n",
       "0    a1000                            Macbook                                                                  Macbook     DD (MIB & Business Analytics)                 25  Female                    ecuador       Hispanic / Latino         Mobile               2.0\n",
       "1    a1001                     Windows laptop                                                           Windows laptop       One year Business Analytics                 27    Male                    Ecuador       Hispanic / Latino         Mobile               2.0\n",
       "2    a1002                     Windows laptop                                                           Windows laptop       One year Business Analytics                 25    Male                     Indian     West Asian / Indian         Online               1.0\n",
       "3    a1003                     Windows laptop                                                           Windows laptop       One year Business Analytics                 26  Female                     indian     West Asian / Indian              0               0.0\n",
       "4    a1004                            Macbook                                                           Windows laptop  DD (MBA & Disruptive innovation)                 27    Male                      China          Far east Asian              0               2.0"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# renaming channels\n",
    "big_5_names = {1 : 'Online',\n",
    "                 2 : 'Mobile'}\n",
    "\n",
    "\n",
    "clst_pca_df['Big 5 Clusters'].replace(big_5_names, inplace = True)\n",
    "\n",
    "clst_pca_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Mobile    53\n",
       "Online    44\n",
       "0.0       41\n",
       "Name: Big 5 Clusters, dtype: int64"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clst_pca_df['Big 5 Clusters'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.0    51\n",
       "3.0    35\n",
       "0.0    31\n",
       "1.0    21\n",
       "Name: Hult_DNA_Cluster, dtype: int64"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clst_pca_df['Hult_DNA_Cluster'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>surveyID</th>\n",
       "      <th>What laptop do you currently have?</th>\n",
       "      <th>What laptop would you buy in next assuming if all laptops cost the same?</th>\n",
       "      <th>What program are you in?</th>\n",
       "      <th>What is your age?</th>\n",
       "      <th>Gender</th>\n",
       "      <th>What is your nationality?</th>\n",
       "      <th>What is your ethnicity?</th>\n",
       "      <th>Big 5 Clusters</th>\n",
       "      <th>Hult_DNA_Cluster</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>a1000</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>DD (MIB &amp; Business Analytics)</td>\n",
       "      <td>25</td>\n",
       "      <td>Female</td>\n",
       "      <td>ecuador</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>Mobile</td>\n",
       "      <td>uu</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>a1001</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>Ecuador</td>\n",
       "      <td>Hispanic / Latino</td>\n",
       "      <td>Mobile</td>\n",
       "      <td>uu</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>a1002</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>25</td>\n",
       "      <td>Male</td>\n",
       "      <td>Indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>Online</td>\n",
       "      <td>yy</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>a1003</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>One year Business Analytics</td>\n",
       "      <td>26</td>\n",
       "      <td>Female</td>\n",
       "      <td>indian</td>\n",
       "      <td>West Asian / Indian</td>\n",
       "      <td>zz</td>\n",
       "      <td>tt</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>a1004</td>\n",
       "      <td>Macbook</td>\n",
       "      <td>Windows laptop</td>\n",
       "      <td>DD (MBA &amp; Disruptive innovation)</td>\n",
       "      <td>27</td>\n",
       "      <td>Male</td>\n",
       "      <td>China</td>\n",
       "      <td>Far east Asian</td>\n",
       "      <td>zz</td>\n",
       "      <td>uu</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  surveyID What laptop do you currently have? What laptop would you buy in next assuming if all laptops cost the same?          What program are you in?  What is your age?  Gender What is your nationality?  What is your ethnicity? Big 5 Clusters Hult_DNA_Cluster\n",
       "0    a1000                            Macbook                                                                  Macbook     DD (MIB & Business Analytics)                 25  Female                    ecuador       Hispanic / Latino         Mobile               uu\n",
       "1    a1001                     Windows laptop                                                           Windows laptop       One year Business Analytics                 27    Male                    Ecuador       Hispanic / Latino         Mobile               uu\n",
       "2    a1002                     Windows laptop                                                           Windows laptop       One year Business Analytics                 25    Male                     Indian     West Asian / Indian         Online               yy\n",
       "3    a1003                     Windows laptop                                                           Windows laptop       One year Business Analytics                 26  Female                     indian     West Asian / Indian             zz               tt\n",
       "4    a1004                            Macbook                                                           Windows laptop  DD (MBA & Disruptive innovation)                 27    Male                      China          Far east Asian             zz               uu"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "big_5_names = {\n",
    "    0 : 'zz',\n",
    "    1 : 'aa',\n",
    "    2 : 'bb',\n",
    "    3 : 'cc',\n",
    "    4 : 'dd',\n",
    "    5 : 'ee'\n",
    "\n",
    "}\n",
    "clst_pca_df['Big 5 Clusters'].replace(big_5_names, inplace = True)\n",
    "\n",
    "\n",
    "hult_dna_names = {\n",
    "    0 : 'tt',\n",
    "    1 : 'yy',\n",
    "    2 : 'uu',\n",
    "    3 : 'ii'\n",
    "}\n",
    "\n",
    "clst_pca_df['Hult_DNA_Cluster'].replace(hult_dna_names, inplace = True)\n",
    "\n",
    "\n",
    "# # renaming regions\n",
    "# cluster_names = {0 : 'Cluster 1',\n",
    "#                  1 : 'Cluster 2',\n",
    "#                  2 : 'Cluster 3'}\n",
    "\n",
    "\n",
    "# final_pca_clust_df['Cluster'].replace(cluster_names, inplace = True)\n",
    "\n",
    "\n",
    "# adding a productivity step\n",
    "data_df = clst_pca_df.copy()\n",
    "\n",
    "\n",
    "# checking results\n",
    "data_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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AIBOBBYAAEAnAgsAAKATgQUAANCJwAIAAOhEYAEAAHQisAAAADoRWAAAAJ0ILAAAgE4EFgAAQCcCCwAAoBOBBQAA0InAAgAA6ERgAQAAdCKwAAAAOhFYAAAAnQgsAACATgQWAABAJwILAACgE4EFAADQicACAADoRGABAAB0IrAAAAA6EVgAAACdCCwAAIBOBBYAAEAnAgsAAKATgQUAANCJwAIAAOhEYAEAAHQisAAAADoRWAAAAJ0ILAAAgE4EFgAAQCcCCwAAoBOBBQAA0InAAgAA6ERgAQAAdCKwAAAAOhFYAAAAnQgsAACATgQWAABAJwILAACgE4EFAADQicACAADoRGABAAB0IrAAAAA6EVgAAACdCCwAAIBOBBYAAEAnAgsAAKATgQUAANCJwAIAAOhEYAEAAHQisAAAADoRWAAAAJ0ILAAAgE4EFgAAQCcCCwAAoBOBBQAA0InAAgAA6ERgAQAAdCKwAAAAOhFYAAAAnQgsAACATgQWAABAJwILAACgE4EFAADQicACAADoRGABAAB0IrAAAAA6EVgAAACdCCwAAIBOBBYAAEAnAgsAAKATgQUAANCJwAIAAOhEYAEAAHQisAAAADoRWAAAAJ0ILAAAgE4EFgAAQCcCCwAAoBOBBQAA0InAAgAA6ERgAQAAdCKwAAAAOhFYAAAAnQgsAACATgQWAABAJwILAACgE4EFAADQicACAADoRGABAAB0IrAAAAA6EVgAAACdCCwAAIBOBBYAAEAnAgsAAKATgQUAANCJwAIAAOhEYAEAAHQisAAAADoRWAAAAJ0ILAAAgE4EFgAAQCcCCwAAoBOBBQAA0InAAgAA6ERgAQAAdCKwAAAAOhFYAAAAnQgsAACATgQWAABAJwILAACgE4EFAADQicACAADoRGABAAB0IrAAAAA6EVgAAACdCCwAAIBOBBYAAEAnAgsAAKATgQUAANCJwAIAAOhEYAEAAHQisAAAADoRWAAAAJ0ILAAAgE4EFgAAQCcCCwAAoBOBBQAA0InAAgAA6ERgAQAAdCKwAAAAOhFYAAAAnQgsAACATgQWAABAJwILAACgE4EFAADQicACAADoRGABAAB0IrAAAAA6EVgAAACdCCwAAIBOBBYAAEAnAgsAAKATgQUAANCJwAIAAOhEYAEAAHQisAAAADoRWAAAAJ0ILAAAgE4EFgAA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\n",
      "text/plain": [
       "<Figure size 864x576 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "########################\n",
    "# Channel\n",
    "########################\n",
    "\n",
    "# Herbivores\n",
    "fig, ax = plt.subplots(figsize = (12, 8))\n",
    "sns.boxplot(x = 'What is your age?',\n",
    "            y = 'Big 5 Clusters',\n",
    "            data = data_df)\n",
    "\n",
    "plt.ylim(-10, 19)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
