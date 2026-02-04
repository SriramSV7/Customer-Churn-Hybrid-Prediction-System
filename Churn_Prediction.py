{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "74f7b9f1-af5e-4bbe-8a18-f9e7099431a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,LabelEncoder\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score,\n",
    "    classification_report,\n",
    "    confusion_matrix,\n",
    "    roc_auc_score)\n",
    "from sklearn.model_selection import GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "033c2384-000b-4744-8f56-15099a141ff8",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv(\"telco_data.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "127f33dd-a50e-45da-9b35-bff3fe154f99",
   "metadata": {
    "scrolled": true
   },
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
       "      <th>customerID</th>\n",
       "      <th>gender</th>\n",
       "      <th>SeniorCitizen</th>\n",
       "      <th>Partner</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>tenure</th>\n",
       "      <th>PhoneService</th>\n",
       "      <th>MultipleLines</th>\n",
       "      <th>InternetService</th>\n",
       "      <th>OnlineSecurity</th>\n",
       "      <th>...</th>\n",
       "      <th>DeviceProtection</th>\n",
       "      <th>TechSupport</th>\n",
       "      <th>StreamingTV</th>\n",
       "      <th>StreamingMovies</th>\n",
       "      <th>Contract</th>\n",
       "      <th>PaperlessBilling</th>\n",
       "      <th>PaymentMethod</th>\n",
       "      <th>MonthlyCharges</th>\n",
       "      <th>TotalCharges</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7590-VHVEG</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>1.0</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>29.85</td>\n",
       "      <td>29.85</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5575-GNVDE</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>34.0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>56.95</td>\n",
       "      <td>1889.5</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3668-QPYBK</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>53.85</td>\n",
       "      <td>108.15</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>7795-CFOCW</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>45.0</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>One year</td>\n",
       "      <td>No</td>\n",
       "      <td>Bank transfer (automatic)</td>\n",
       "      <td>42.30</td>\n",
       "      <td>1840.75</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9237-HQITU</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>70.70</td>\n",
       "      <td>151.65</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7038</th>\n",
       "      <td>6840-RESVB</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>24.0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>One year</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>84.80</td>\n",
       "      <td>1990.5</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7039</th>\n",
       "      <td>2234-XADUH</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>72.0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>One year</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Credit card (automatic)</td>\n",
       "      <td>103.20</td>\n",
       "      <td>7362.9</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7040</th>\n",
       "      <td>4801-JZAZL</td>\n",
       "      <td>Female</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>11.0</td>\n",
       "      <td>No</td>\n",
       "      <td>No phone service</td>\n",
       "      <td>DSL</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Electronic check</td>\n",
       "      <td>29.60</td>\n",
       "      <td>346.45</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7041</th>\n",
       "      <td>8361-LTMKD</td>\n",
       "      <td>Male</td>\n",
       "      <td>1</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>No</td>\n",
       "      <td>...</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>Month-to-month</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Mailed check</td>\n",
       "      <td>74.40</td>\n",
       "      <td>306.6</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7042</th>\n",
       "      <td>3186-AJIEK</td>\n",
       "      <td>Male</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "      <td>66.0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>No</td>\n",
       "      <td>Fiber optic</td>\n",
       "      <td>Yes</td>\n",
       "      <td>...</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Two year</td>\n",
       "      <td>Yes</td>\n",
       "      <td>Bank transfer (automatic)</td>\n",
       "      <td>105.65</td>\n",
       "      <td>6844.5</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>7043 rows × 21 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      customerID  gender  SeniorCitizen Partner Dependents  tenure  \\\n",
       "0     7590-VHVEG  Female              0     Yes         No     1.0   \n",
       "1     5575-GNVDE    Male              0      No         No    34.0   \n",
       "2     3668-QPYBK    Male              0      No         No     2.0   \n",
       "3     7795-CFOCW    Male              0      No         No    45.0   \n",
       "4     9237-HQITU  Female              0      No         No     2.0   \n",
       "...          ...     ...            ...     ...        ...     ...   \n",
       "7038  6840-RESVB    Male              0     Yes        Yes    24.0   \n",
       "7039  2234-XADUH  Female              0     Yes        Yes    72.0   \n",
       "7040  4801-JZAZL  Female              0     Yes        Yes    11.0   \n",
       "7041  8361-LTMKD    Male              1     Yes         No     NaN   \n",
       "7042  3186-AJIEK    Male              0      No         No    66.0   \n",
       "\n",
       "     PhoneService     MultipleLines InternetService OnlineSecurity  ...  \\\n",
       "0              No  No phone service             DSL             No  ...   \n",
       "1             Yes                No             DSL            Yes  ...   \n",
       "2             Yes                No             DSL            Yes  ...   \n",
       "3              No  No phone service             DSL            Yes  ...   \n",
       "4             Yes                No     Fiber optic             No  ...   \n",
       "...           ...               ...             ...            ...  ...   \n",
       "7038          Yes               Yes             DSL            Yes  ...   \n",
       "7039          Yes               Yes     Fiber optic             No  ...   \n",
       "7040           No  No phone service             DSL            Yes  ...   \n",
       "7041          Yes               Yes     Fiber optic             No  ...   \n",
       "7042          Yes                No     Fiber optic            Yes  ...   \n",
       "\n",
       "     DeviceProtection TechSupport StreamingTV StreamingMovies        Contract  \\\n",
       "0                  No          No          No              No  Month-to-month   \n",
       "1                 Yes          No          No              No        One year   \n",
       "2                  No          No          No              No  Month-to-month   \n",
       "3                 Yes         Yes          No              No        One year   \n",
       "4                  No          No          No              No  Month-to-month   \n",
       "...               ...         ...         ...             ...             ...   \n",
       "7038              Yes         Yes         Yes             Yes        One year   \n",
       "7039              Yes          No         Yes             Yes        One year   \n",
       "7040               No          No          No              No  Month-to-month   \n",
       "7041               No          No          No              No  Month-to-month   \n",
       "7042              Yes         Yes         Yes             Yes        Two year   \n",
       "\n",
       "     PaperlessBilling              PaymentMethod MonthlyCharges  TotalCharges  \\\n",
       "0                 Yes           Electronic check          29.85         29.85   \n",
       "1                  No               Mailed check          56.95        1889.5   \n",
       "2                 Yes               Mailed check          53.85        108.15   \n",
       "3                  No  Bank transfer (automatic)          42.30       1840.75   \n",
       "4                 Yes           Electronic check          70.70        151.65   \n",
       "...               ...                        ...            ...           ...   \n",
       "7038              Yes               Mailed check          84.80        1990.5   \n",
       "7039              Yes    Credit card (automatic)         103.20        7362.9   \n",
       "7040              Yes           Electronic check          29.60        346.45   \n",
       "7041              Yes               Mailed check          74.40         306.6   \n",
       "7042              Yes  Bank transfer (automatic)         105.65        6844.5   \n",
       "\n",
       "     Churn  \n",
       "0       No  \n",
       "1       No  \n",
       "2      Yes  \n",
       "3       No  \n",
       "4      Yes  \n",
       "...    ...  \n",
       "7038    No  \n",
       "7039    No  \n",
       "7040    No  \n",
       "7041   Yes  \n",
       "7042    No  \n",
       "\n",
       "[7043 rows x 21 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0745d8b0-d148-465a-b9df-f24b05ec64fa",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 7043 entries, 0 to 7042\n",
      "Data columns (total 20 columns):\n",
      " #   Column            Non-Null Count  Dtype  \n",
      "---  ------            --------------  -----  \n",
      " 0   gender            6293 non-null   object \n",
      " 1   SeniorCitizen     7043 non-null   int64  \n",
      " 2   Partner           6043 non-null   object \n",
      " 3   Dependents        7043 non-null   object \n",
      " 4   tenure            4543 non-null   float64\n",
      " 5   PhoneService      7043 non-null   object \n",
      " 6   MultipleLines     7043 non-null   object \n",
      " 7   InternetService   6043 non-null   object \n",
      " 8   OnlineSecurity    7043 non-null   object \n",
      " 9   OnlineBackup      7043 non-null   object \n",
      " 10  DeviceProtection  7043 non-null   object \n",
      " 11  TechSupport       7043 non-null   object \n",
      " 12  StreamingTV       5543 non-null   object \n",
      " 13  StreamingMovies   7043 non-null   object \n",
      " 14  Contract          7043 non-null   object \n",
      " 15  PaperlessBilling  7043 non-null   object \n",
      " 16  PaymentMethod     7043 non-null   object \n",
      " 17  MonthlyCharges    5543 non-null   float64\n",
      " 18  TotalCharges      7043 non-null   object \n",
      " 19  Churn             7043 non-null   object \n",
      "dtypes: float64(2), int64(1), object(17)\n",
      "memory usage: 1.1+ MB\n"
     ]
    }
   ],
   "source": [
    "### Information about dataset\n",
    "### Drop the Customer ID because it doesnt realtable to the Churn\n",
    "df=df.drop(columns=\"customerID\")\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9a75729d-fb36-4890-ac62-48805475a2f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "### finding missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b70462f5-8b60-4c43-80bb-effe523011be",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)\n",
    "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
    "df.dropna(subset=['TotalCharges'], inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b7f4a372-d773-4d4a-a828-ed200b42aa76",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 7032 entries, 0 to 7042\n",
      "Data columns (total 20 columns):\n",
      " #   Column            Non-Null Count  Dtype  \n",
      "---  ------            --------------  -----  \n",
      " 0   gender            6282 non-null   object \n",
      " 1   SeniorCitizen     7032 non-null   int64  \n",
      " 2   Partner           6033 non-null   object \n",
      " 3   Dependents        7032 non-null   object \n",
      " 4   tenure            4535 non-null   float64\n",
      " 5   PhoneService      7032 non-null   object \n",
      " 6   MultipleLines     7032 non-null   object \n",
      " 7   InternetService   6033 non-null   object \n",
      " 8   OnlineSecurity    7032 non-null   object \n",
      " 9   OnlineBackup      7032 non-null   object \n",
      " 10  DeviceProtection  7032 non-null   object \n",
      " 11  TechSupport       7032 non-null   object \n",
      " 12  StreamingTV       5534 non-null   object \n",
      " 13  StreamingMovies   7032 non-null   object \n",
      " 14  Contract          7032 non-null   object \n",
      " 15  PaperlessBilling  7032 non-null   object \n",
      " 16  PaymentMethod     7032 non-null   object \n",
      " 17  MonthlyCharges    5534 non-null   float64\n",
      " 18  TotalCharges      7032 non-null   float64\n",
      " 19  Churn             7032 non-null   object \n",
      "dtypes: float64(3), int64(1), object(16)\n",
      "memory usage: 1.1+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2aa55be0-d295-48ba-9a9f-51f5191432f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "gender               750\n",
       "SeniorCitizen          0\n",
       "Partner              999\n",
       "Dependents             0\n",
       "tenure              2497\n",
       "PhoneService           0\n",
       "MultipleLines          0\n",
       "InternetService      999\n",
       "OnlineSecurity         0\n",
       "OnlineBackup           0\n",
       "DeviceProtection       0\n",
       "TechSupport            0\n",
       "StreamingTV         1498\n",
       "StreamingMovies        0\n",
       "Contract               0\n",
       "PaperlessBilling       0\n",
       "PaymentMethod          0\n",
       "MonthlyCharges      1498\n",
       "TotalCharges           0\n",
       "Churn                  0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "8b4370d2-7c54-4ed9-9924-d25c4989e37a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Female' 'Male' nan]\n",
      "['Yes' 'No' nan]\n",
      "[ 1. 34.  2. 45.  8. 22. 10. nan 62. 13. 16. 49. 52. 12. 30. 72. 27.  5.\n",
      " 11. 17. 69. 25. 60. 18. 63. 47.  9. 31. 50. 64.  3. 46.  7. 42. 35. 29.\n",
      " 66. 68. 43. 36. 41. 56. 33. 71. 23. 57. 65. 70.  6. 14. 20. 53.  4. 32.\n",
      " 15. 40. 59. 44. 61. 24. 51. 67. 19. 21. 48. 38. 54. 28. 58. 37. 55. 39.\n",
      " 26.]\n",
      "['DSL' 'Fiber optic' 'No' nan]\n",
      "['No' 'Yes' nan 'No internet service']\n",
      "[29.85 56.95 53.85 ... 60.4  44.2  78.7 ]\n"
     ]
    }
   ],
   "source": [
    "### Filling null values for object by mode and numbers by mean\n",
    "print(df[\"gender\"].unique())\n",
    "print(df[\"Partner\"].unique())\n",
    "print(df[\"tenure\"].unique())\n",
    "print(df[\"InternetService\"].unique())\n",
    "print(df[\"StreamingTV\"].unique())\n",
    "print(df[\"MonthlyCharges\"].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b4001ba8-6213-493b-b75e-9c52e741a720",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"gender\"] = df[\"gender\"].fillna(df[\"gender\"].mode()[0])\n",
    "df[\"Partner\"] = df[\"Partner\"].fillna(df[\"Partner\"].mode()[0])\n",
    "df[\"tenure\"] = df[\"tenure\"].fillna(df[\"tenure\"].median())\n",
    "df[\"InternetService\"] = df[\"InternetService\"].fillna(df[\"InternetService\"].mode()[0])\n",
    "df[\"StreamingTV\"] = df[\"StreamingTV\"].fillna(df[\"StreamingTV\"].mode()[0])\n",
    "df[\"MonthlyCharges\"] = df[\"MonthlyCharges\"].fillna(df[\"MonthlyCharges\"].mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6e599d12-a7d6-4497-842f-dac889948505",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "gender              0\n",
       "SeniorCitizen       0\n",
       "Partner             0\n",
       "Dependents          0\n",
       "tenure              0\n",
       "PhoneService        0\n",
       "MultipleLines       0\n",
       "InternetService     0\n",
       "OnlineSecurity      0\n",
       "OnlineBackup        0\n",
       "DeviceProtection    0\n",
       "TechSupport         0\n",
       "StreamingTV         0\n",
       "StreamingMovies     0\n",
       "Contract            0\n",
       "PaperlessBilling    0\n",
       "PaymentMethod       0\n",
       "MonthlyCharges      0\n",
       "TotalCharges        0\n",
       "Churn               0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2d626e6d-03ec-4ea0-a011-33c1123d1bd5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Female' 'Male']\n",
      "['Yes' 'No']\n",
      "[ 1. 34.  2. 45.  8. 22. 10. 29. 62. 13. 16. 49. 52. 12. 30. 72. 27.  5.\n",
      " 11. 17. 69. 25. 60. 18. 63. 47.  9. 31. 50. 64.  3. 46.  7. 42. 35. 66.\n",
      " 68. 43. 36. 41. 56. 33. 71. 23. 57. 65. 70.  6. 14. 20. 53.  4. 32. 15.\n",
      " 40. 59. 44. 61. 24. 51. 67. 19. 21. 48. 38. 54. 28. 58. 37. 55. 39. 26.]\n",
      "['DSL' 'Fiber optic' 'No']\n",
      "['No' 'Yes' 'No internet service']\n",
      "[29.85 56.95 53.85 ... 60.4  44.2  78.7 ]\n"
     ]
    }
   ],
   "source": [
    "print(df[\"gender\"].unique())\n",
    "print(df[\"Partner\"].unique())\n",
    "print(df[\"tenure\"].unique())\n",
    "print(df[\"InternetService\"].unique())\n",
    "print(df[\"StreamingTV\"].unique())\n",
    "print(df[\"MonthlyCharges\"].unique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "53dc95fc-e231-4837-bb3d-e0a263bce0c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "gender              0\n",
       "SeniorCitizen       0\n",
       "Partner             0\n",
       "Dependents          0\n",
       "tenure              0\n",
       "PhoneService        0\n",
       "MultipleLines       0\n",
       "InternetService     0\n",
       "OnlineSecurity      0\n",
       "OnlineBackup        0\n",
       "DeviceProtection    0\n",
       "TechSupport         0\n",
       "StreamingTV         0\n",
       "StreamingMovies     0\n",
       "Contract            0\n",
       "PaperlessBilling    0\n",
       "PaymentMethod       0\n",
       "MonthlyCharges      0\n",
       "TotalCharges        0\n",
       "Churn               0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### finding duplicated rows\n",
    "\n",
    "df.drop_duplicates().isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "518fe315-c209-47bb-89c1-42913cdb1c64",
   "metadata": {},
   "outputs": [],
   "source": [
    "### separates it as two columns\n",
    "num_col = df[[\"SeniorCitizen\", \"tenure\", \"MonthlyCharges\", \"TotalCharges\"]]\n",
    "# Fix: Use a single list of column names instead of a list of lists\n",
    "#cat_col = df.drop(columns=[\"SeniorCitizen\", \"tenure\", \"MonthlyCharges\", \"TotalCharges\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "b8dab52f-d1a8-4866-abf8-3c49f908952a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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z549jTHGFBUVmbFjx5pOnTpZ61pUVGSMORECmjdvbpYtW2b++9//mtGjR5vQ0NBKt13xxJWUlGRuvPFGt7nceuut1hPPjBkzjCS3MHI6FY8P/fr1M5s2bTKZmZmmQ4cOZujQoVbNxx9/bBYvXmy2bt1qtm7daoYPH24iIyPdXlxJMhEREebVV181u3btMnv27DFLliwxgYGB5rXXXjPZ2dlm4sSJJjw83O2xYt68eaZZs2bm7bffNt9//715++23TaNGjczChQuNMcZMmzbNxMTEmM8++8zs2bPHfP7552bZsmVVWtvz5dQQs2LFCuPv729eeOEFk52dbaZPn258fX3NJ598Yow58+NkVlaWefnll823335rduzYYR5//HETGBho9u7da932yY+3K1asMJJMRkbGWcd3ruPOk77Dw8PNtGnTzH//+1/z3//+16xdu9Y0aNDATJs2zWRnZ5sXXnjBNGrUyG1/pKammvDwcLNw4UKza9cuk5aWZlq2bGlSUlKMMca89dZbJjw83KxcudLs3bvXbNiwwcybN69aa/FbEGKq4F//+pdp2LChCQwMND179jQTJkww33zzjTHmxANBeHh4pYBz2WWXmblz5xpjToSBildSFcaPH29iY2OtyyeHmB07dhhJ5osvvrCuP3z4sAkKCjL//Oc/jTH//6u/k8/4HDx40EgyM2bMOOecTr5TGXPiweqdd95xqzk1xFQoLy83t956q+nWrZv1YP3kk0+a+Ph4t7p9+/YZSSY7O9ua49VXX+1Wc9VVV5m//vWv5xxvfXFqWD0f+82TEPPuu+9aNYWFhSYwMLDSg+nw4cPNnXfe+VunWqcqQsyhQ4eMw+Ewu3fvNnv27DGBgYHm0KFDVogpLCw0/v7+ZunSpVbb0tJSEx0dbaZOnWqMcX9FXOHf//63kWSKi4uNMWe+r0gyTzzxhHW5sLDQ+Pj4mI8++sjttitCzIYNG4yvr691JvXQoUPG39/fCpV//vOfTXh4+DnnX/H4sHPnTmvbCy+8YCIjI8/YpqyszISFhZkPPvjAbfynnvWJjY01Dz74oNu2Xr16uc0/JiamUih59tlnTVxcnDHGmFGjRpn/+Z//MeXl5eecS005NcT07NnTJCUludUMHjzYLVSe7nHydDp27Ghmz55tXT758XbKlClGkvnll1/OehtVOe6q2vepZ33uuOMOc9NNN7ltu+uuu9z2xzXXXFPp7NjixYtNs2bNjDHGTJ8+3bRt29Z6YV1X+ExMFdx+++3av3+/3n//fSUkJGj16tXq2rWrFi5cqMzMTBUWFqpx48YKDQ21/u3evVu7du2ybqNly5Zun3lp1qyZcnNzT9vftm3b5Ofnp9jYWGtb48aN1a5dO23bts3aFhAQoMsvv9y6bP7fh3Z9fHzO29xP57HHHtO6dev07rvvKigoSJKUmZmpTz/91G0ftG/fXpLc9sPJ45XOvh+8QW3vt+7du1v/37p1q3799Vf179/frf/XX3/drW87a9KkiW666SYtWrRICxYs0E033aQmTZpY1+/atUsul0u9evWytvn7++v3v/+9231Ncl+DZs2aSVKV1uDkdiEhIQoLCztju9///vfq1KmTXn/9dUnS4sWLdckll6h3796STtzHq3r/Dg4O1mWXXeY25pP7zc3N1Z/+9Ce1bdtWTqdTTqdThYWF+uGHH9xu5+RjRpKys7P1+9//vtK4Kxw6dEj79u3T8OHD3Y6r5557zjqu7r33XmVlZaldu3YaPXq00tLSqjSnmrRt2za340CSevXqVek4ONWxY8f0yCOPqGPHjrrooosUGhqq7du3V9qPFSoep6vqbMddVfv2dA2lE49NzzzzjNsaJiUl6cCBAyoqKtLgwYNVXFysSy+9VElJSXrnnXfq5HN2frXeo00FBgaqf//+6t+/v5566indf//9evrppzVy5Eg1a9ZMq1evrtTm5K84+vv7u13n4+Oj8vLy0/Z1poP81AewoKAgt8tNmzZVw4YNz3mn+y2WLFmimTNnavXq1WrevLm1vby8XDfffLOmTJlSqU3FHU/ybD94g/Ox3xo0OPFa5OTj5kwfwAwJCXHrW5L+/e9/6+KLL3arq09/q+W+++7TQw89JEl64YUX3K47U/A/XVg4eQ0qrqvKsevpMX///fdrzpw5evTRR7VgwQL97//+r9Vf27ZtlZ+frwMHDrgdH1Xt9+Rj5N5779WhQ4c0a9YstWjRQg6HQ3FxcZU++HnyMXPybZ3s5NutmNv8+fPdXohJkq+vrySpa9eu2r17tz766COtWrVKQ4YMUb9+/fSvf/3rrHOqaVU5Dk41fvx4/ec//9Hf//53tW7dWkFBQfrDH/5wxg/Qtm3bVpK0fft2xcXFnXNMZzvuqtr3qWt4unmd+rxTXl6uiRMn6rbbbqs0psDAQMXExCg7O1vp6elatWqVRo4cqWnTpmnNmjWVjr2axJmYaurYsaOOHTumrl27KicnR35+fmrdurXbv5Nf8Xl622VlZdqwYYO17eeff9aOHTvUoUOHM7Zr0KCB7rjjDi1dulT79++vdP2xY8fOmJT9/f11/Pjxs45r3bp1uv/++zV37lz16NHD7bquXbtqy5YtatmyZaX9cLoHQW8VEBDgtp/Px35r2rSppBPfvKiQlZV1znYdO3aUw+HQDz/8UKnvmJgYzyZ2Abv++utVWlqq0tJSJSQkuF3XunVrBQQEaO3atdY2l8ulL7/88qz3tVOduq6/xd13360ffvhB//jHP7RlyxYNGzbMuu4Pf/iDAgICNHXq1NO29eR3inz++ecaPXq0brzxRnXq1EkOh0OHDx8+Z7t27dpp48aNbtu+/PJL6/+RkZG6+OKL9f3331c6rlq1amXVhYeH64477tD8+fP15ptv6u233z6v3/j0VIcOHdyOA0nKyMhwOw5O9zj5+eef695779Wtt96qLl26KCoqSnv27DljP/Hx8WrSpMl5W0NP+q7Qvn37s66hdOKxKTs7u9Iatm7d2nrhFBQUpIEDB+of//iHVq9erXXr1mnz5s1VHv/5wJmYc/j55581ePBg3Xfffbr88ssVFhamL7/8UlOnTtUtt9yifv36KS4uToMGDdKUKVPUrl077d+/XytXrtSgQYMqncarijZt2uiWW25RUlKS5s6dq7CwMD366KO6+OKLdcstt5y17aRJk7R69WrFxsbq+eefV/fu3eXv76/PP/9ckydP1qZNm077S7Batmypjz/+WL169ZLD4VDDhg3drs/JydGtt96qP/7xj0pISFBOTo6kE6+smjZtqgcffFDz58/XnXfeqfHjx6tJkybauXOnli9frvnz51uvwLxdy5YttWHDBu3Zs0ehoaHnZb8FBQWpR48e+tvf/qaWLVvq8OHDeuKJJ87ZLiwsTOPGjdNf/vIXlZeX6+qrr1ZBQYEyMjIUGhrq9uRpZ76+vtbZyVP3Z0hIiP785z9r/PjxatSokS655BJNnTpVRUVFGj58eJX7aNmypXbv3q2srCw1b95cYWFh1T6b1bBhQ912220aP3684uPj3c54xsTEaObMmXrooYdUUFCge+65Ry1bttSPP/6o119/XaGhoVX+mnXr1q21ePFide/eXQUFBRo/frz19vDZjBo1SklJSerevbt69uypN998U99++60uvfRSqyYlJUWjR49WeHi4brjhBpWUlOjLL79UXl6exowZo5kzZ6pZs2a64oor1KBBA7311luKioqq01/QN378eA0ZMkRdu3ZV37599cEHH2jFihVatWqVVXO6x8nWrVtrxYoVuvnmm+Xj46Mnn3zyrGfaQkJC9Morr2jw4MEaOHCgRo8erdatW+vw4cP65z//qR9++EHLly+v0pg97bvCqFGj1Lt3b82YMUM333yzPvnkE3300UduZ2eeeuopDRgwQDExMRo8eLAaNGigb7/9Vps3b9Zzzz2nhQsX6vjx44qNjVVwcLAWL16soKAgtWjRokpjP184E3MOoaGhio2N1cyZM9W7d2917txZTz75pJKSkjRnzhz5+Pho5cqV6t27t+677z61bdtWf/zjH7Vnzx5FRkZWu98FCxaoW7duGjBggOLi4mSM0cqVK895mq5hw4Zav3697r77bj333HO68sordc011+iNN97QtGnT5HQ6T9tu+vTpSk9PV0xMjK688spK12/fvl0HDx7UokWL1KxZM+vfVVddJUmKjo7WF198oePHjyshIUGdO3fWww8/LKfTaaV2nPgFgr6+vurYsaOaNm2q0tLS87LfXnvtNblcLnXv3l0PP/ywnnvuuSq1e/bZZ/XUU09p8uTJ6tChgxISEvTBBx+4vWKuD8LDwxUeHn7a6/72t7/p9ttvV2Jiorp27aqdO3fqP//5T6Ugfza33367rr/+el133XVq2rSp3njjjd803uHDh6u0tFT33XdfpetGjhyptLQ0/fTTT7r11lvVvn173X///QoPD9e4ceOq3Mdrr72mvLw8XXnllUpMTNTo0aMVERFxznZ33XWXJkyYoHHjxllvC917770KDAy0au6//3698sorWrhwobp06aJrr71WCxcutI6r0NBQTZkyRd27d9dVV12lPXv2aOXKlXX6WDFo0CD93//9n6ZNm6ZOnTpp7ty5WrBggfr06WPVnO5xcubMmWrYsKF69uypm2++WQkJCeratetZ+7rllluUkZEhf39/DR06VO3bt9edd96p/Pz8Kt93q9u3dOKzPi+//LJmzJih3/3ud0pNTdVf/vIXtzVMSEjQhx9+qPT0dF111VXq0aOHZsyYYYWUiy66SPPnz1evXr10+eWX6+OPP9YHH3xQ679bzMd4+ikjAECNWrp0qR5++GHt379fAQEBdT2cc+rfv7+ioqK0ePHiuh4KqikpKUnbt2/X559/XtdD8QhvJwHABaKoqEi7d+/W5MmTNWLEiAsywBQVFenll19WQkKCfH199cYbb2jVqlVKT0+v66HBA3//+9/Vv39/hYSE6KOPPtKiRYv04osv1vWwPMaZGAC4QKSkpOj5559X79699d577yk0NLSuh1RJcXGxbr75Zn311VcqKSlRu3bt9MQTT5z2Wyy4cA0ZMkSrV6/W0aNHdemll2rUqFGn/VtjFzpCDAAAsCU+cQkAAGyJEAMAAGyJEAMAAGyJEAMAAGyJEAMAAGyJEAMAAGyJEAMAAGyJEAMAAGyJEAMAAGzp/wPjcT3iI9uf8wAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "### Finding Outliers\n",
    "num_col.boxplot()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7fd923e4-4ac3-49a4-ad3b-0447e42d135a",
   "metadata": {},
   "outputs": [],
   "source": [
    "### There is no outliers present in the  num_col \n",
    "### the seniorcitizen colunm only having two numbers(0,1) it is discrete colunm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2ef36f16-15b1-4b7a-bdaf-15c994d77f16",
   "metadata": {},
   "outputs": [],
   "source": [
    "### one hot encoder for labelling text into numbers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "b7902913-f237-4a9f-9e58-cbb721ff15d9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Index: 7032 entries, 0 to 7042\n",
      "Data columns (total 20 columns):\n",
      " #   Column            Non-Null Count  Dtype  \n",
      "---  ------            --------------  -----  \n",
      " 0   gender            7032 non-null   object \n",
      " 1   SeniorCitizen     7032 non-null   int64  \n",
      " 2   Partner           7032 non-null   object \n",
      " 3   Dependents        7032 non-null   object \n",
      " 4   tenure            7032 non-null   float64\n",
      " 5   PhoneService      7032 non-null   object \n",
      " 6   MultipleLines     7032 non-null   object \n",
      " 7   InternetService   7032 non-null   object \n",
      " 8   OnlineSecurity    7032 non-null   object \n",
      " 9   OnlineBackup      7032 non-null   object \n",
      " 10  DeviceProtection  7032 non-null   object \n",
      " 11  TechSupport       7032 non-null   object \n",
      " 12  StreamingTV       7032 non-null   object \n",
      " 13  StreamingMovies   7032 non-null   object \n",
      " 14  Contract          7032 non-null   object \n",
      " 15  PaperlessBilling  7032 non-null   object \n",
      " 16  PaymentMethod     7032 non-null   object \n",
      " 17  MonthlyCharges    7032 non-null   float64\n",
      " 18  TotalCharges      7032 non-null   float64\n",
      " 19  Churn             7032 non-null   object \n",
      "dtypes: float64(3), int64(1), object(16)\n",
      "memory usage: 1.1+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "raw",
   "id": "3294d46c-c6f9-4bb0-96ce-f9495d4d77a3",
   "metadata": {},
   "source": [
    "cat_col= df.select_dtypes(include=['object']).columns.tolist()\n",
    "\n",
    "# 2. Apply One-Hot Encoding\n",
    "# pd.get_dummies handles index alignment, column naming, and conversion automatically.\n",
    "# drop_first=True prevents multicollinearity.\n",
    "df_encoded = pd.get_dummies(df, columns=cat_col, drop_first=True)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "7e9e4d61-4021-4f1a-8930-132b6a2e17ea",
   "metadata": {},
   "source": [
    "df=df_encoded"
   ]
  },
  {
   "cell_type": "raw",
   "id": "363c6235-9404-470b-bd08-1d3c58f4e108",
   "metadata": {},
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "raw",
   "id": "83eae956-ca06-4d03-b27f-20b89cadbae9",
   "metadata": {},
   "source": [
    "### Feature Selection\n",
    "corr=df.corr()\n",
    "sns.heatmap(corr,annot=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "raw",
   "id": "d0d34df8-2857-4530-aacf-c06421da1075",
   "metadata": {},
   "source": [
    "from sklearn.feature_selection import f_classif\n",
    "\n",
    "target=df[\"Churn_Yes\"]\n",
    "feature=df.drop(columns=[\"Churn_Yes\"])\n",
    "\n",
    "f_clas=f_classif(feature,target)\n",
    "pd.Series(f_clas[0],index=feature.columns).sort_values(ascending=False).plot(kind='bar')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "raw",
   "id": "7585a081-8d63-446c-9c6f-cda3fb24169f",
   "metadata": {},
   "source": [
    "feature_numeric = feature.select_dtypes(include=[np.number]).copy()\n",
    "\n",
    "# Check for and handle NaN values\n",
    "feature_numeric = feature_numeric.fillna(0)  # Replace NaN with 0 or use other methods like mean/median\n",
    "\n",
    "# Calculate VIF\n",
    "vif = pd.DataFrame()\n",
    "vif['Features'] = feature_numeric.columns\n",
    "vif['VIF'] = [variance_inflation_factor(feature_numeric.values, i) for i in range(len(feature_numeric.columns))]\n",
    "vif.sort_values(by='VIF', ascending=False)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "623a673a-6305-4d50-8dfd-0241e9c02aab",
   "metadata": {},
   "source": [
    "std_sca=StandardScaler()"
   ]
  },
  {
   "cell_type": "raw",
   "id": "ce88c22b-3d59-4f95-b455-26ecac097c55",
   "metadata": {},
   "source": [
    "num_cols=std_sca.fit_transform(num_col)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "b9edb6bc-01c8-46d1-ac8c-563a4c342170",
   "metadata": {},
   "source": [
    "num_cols.shape"
   ]
  },
  {
   "cell_type": "raw",
   "id": "87101388-1a75-491a-927e-d5d2c32f6cbc",
   "metadata": {},
   "source": [
    "num_cols"
   ]
  },
  {
   "cell_type": "raw",
   "id": "4a6eb3dc-d691-416c-a54e-11bf4001e153",
   "metadata": {},
   "source": [
    "num_cols_df = pd.DataFrame(num_cols)  # You might need to specify column names here\n",
    "df=pd.concat([num_cols_df,df_encoded], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "64c8a77c-346c-472e-b52c-20bb55f5d6ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical_cols = df.select_dtypes(include=[\"object\"]).columns\n",
    "\n",
    "label_encoders = {}\n",
    "\n",
    "for col in categorical_cols:\n",
    "    le = LabelEncoder()\n",
    "    df[col] = le.fit_transform(df[col])\n",
    "    label_encoders[col] = le"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "aa9882f9-ddc4-4fae-996a-fbcca8e2a33c",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(\"Churn\", axis=1)\n",
    "y = df[\"Churn\"]\n",
    "\n",
    "# ===============================\n",
    "# 7. TRAIN / TEST SPLIT\n",
    "# ===============================\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X,\n",
    "    y,\n",
    "    test_size=0.2,\n",
    "    random_state=150,\n",
    "    stratify=y\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "d40918bb-3745-4001-a1e5-5ed6935abc18",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(5625, 19)\n",
      "(1407, 19)\n",
      "(5625,)\n",
      "(1407,)\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape)\n",
    "print(X_test.shape)\n",
    "print(y_train.shape)\n",
    "print(y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "c4c12650-b02e-4d2d-88e5-26ea1231b3cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7910447761194029"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rand_for= RandomForestClassifier(n_estimators=100,criterion='gini', max_features='sqrt',\n",
    "                                 bootstrap=True)\n",
    "rand_for.fit(X_train,y_train)\n",
    "y_pred= rand_for.predict(X_test)\n",
    "accuracy_score(y_test,y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "1f9ee0f4-3638-4619-9892-5858ff772b7a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'max_depth': 10, 'n_estimators': 100}"
      ]
     },
     "execution_count": 57,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "params={'n_estimators':[10,50,100,150,200],\"max_depth\":[10,15,20,25,30]}\n",
    "grid_search=GridSearchCV(rand_for,params,cv=5)\n",
    "grid_search.fit(X_train,y_train)\n",
    "grid_search.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "a16d3749-951c-4b50-a9dd-2c8048888466",
   "metadata": {},
   "outputs": [],
   "source": [
    "best_rf = grid_search.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "858885ff-db1f-4ac6-a09c-4f69ba0a7cef",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7910447761194029"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rand_for=RandomForestClassifier(max_depth=10,n_estimators=100)\n",
    "rand_for.fit(X_train,y_train)\n",
    "y_pred=rand_for.predict(X_test)\n",
    "accuracy_score(y_test,y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "3faa368e-1fe7-4aea-bb5b-7f7bf7b7fd92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.7910447761194029\n",
      "\n",
      "Confusion Matrix:\n",
      " [[923 110]\n",
      " [184 190]]\n",
      "\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.83      0.89      0.86      1033\n",
      "           1       0.63      0.51      0.56       374\n",
      "\n",
      "    accuracy                           0.79      1407\n",
      "   macro avg       0.73      0.70      0.71      1407\n",
      "weighted avg       0.78      0.79      0.78      1407\n",
      "\n",
      "ROC-AUC Score: 0.8410773356249127\n"
     ]
    }
   ],
   "source": [
    "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(\"\\nConfusion Matrix:\\n\", confusion_matrix(y_test, y_pred))\n",
    "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n",
    "print(\"ROC-AUC Score:\", roc_auc_score(y_test, y_prob))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "e215831b-6d76-41e2-9f8b-25573540c2c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "56a5ca5f-9131-4f2a-b419-4c2fb10fd4bc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['feature_columns.pkl']"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "\n",
    "joblib.dump(best_rf, \"rf_churn_model.pkl\")\n",
    "joblib.dump(X.columns.tolist(), \"feature_columns.pkl\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "c8ade569-1847-4c18-ae3b-4c7fad89b3f8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-12-13 15:32:06.325 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.326 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.372 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /Applications/Anaconda/anaconda3/lib/python3.13/site-packages/ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-12-13 15:32:06.373 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.373 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.374 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.374 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.375 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.376 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.378 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.378 Session state does not function when running a script without `streamlit run`\n",
      "2025-12-13 15:32:06.379 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.379 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.380 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.380 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.380 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.381 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.381 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.382 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.382 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.383 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.383 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.384 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.385 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.385 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.390 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.395 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.396 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.397 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.398 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.398 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.399 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.399 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.401 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.401 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.405 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.407 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.408 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.412 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.422 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.423 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.424 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.425 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.426 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.427 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.427 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.428 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.428 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.429 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.429 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.430 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.430 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.431 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.431 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.432 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.432 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.433 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.433 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.434 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.434 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.435 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.438 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.440 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.440 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.441 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.441 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.442 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.443 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.443 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.444 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.444 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.445 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.446 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.447 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.448 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.448 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.449 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.449 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.449 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.450 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.451 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.451 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.452 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.454 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.455 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.455 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.456 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.457 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.457 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.458 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.458 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.459 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.460 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.460 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.461 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.462 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.462 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.462 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.463 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.463 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.464 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.465 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.466 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.466 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.467 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.470 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.471 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.472 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.473 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.474 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.475 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.476 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.476 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.476 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.477 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.477 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.478 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.478 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.479 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.495 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.496 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.496 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.497 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-12-13 15:32:06.497 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# ===============================\n",
    "# LOAD MODEL & METADATA\n",
    "# ===============================\n",
    "model = joblib.load(\"rf_churn_model.pkl\")\n",
    "feature_columns = joblib.load(\"feature_columns.pkl\")\n",
    "\n",
    "st.set_page_config(page_title=\"Telco Churn Predictor\", layout=\"centered\")\n",
    "\n",
    "st.title(\"📊 Telco Customer Churn Prediction\")\n",
    "st.write(\"Predict whether a customer is likely to churn using a Random Forest model.\")\n",
    "\n",
    "# ===============================\n",
    "# USER INPUT FORM\n",
    "# ===============================\n",
    "st.subheader(\"Customer Information\")\n",
    "\n",
    "def user_input():\n",
    "    gender = st.selectbox(\"Gender\", [\"Male\", \"Female\"])\n",
    "    SeniorCitizen = st.selectbox(\"Senior Citizen\", [0, 1])\n",
    "    Partner = st.selectbox(\"Partner\", [\"Yes\", \"No\"])\n",
    "    Dependents = st.selectbox(\"Dependents\", [\"Yes\", \"No\"])\n",
    "    tenure = st.slider(\"Tenure (months)\", 0, 72, 12)\n",
    "    PhoneService = st.selectbox(\"Phone Service\", [\"Yes\", \"No\"])\n",
    "    MultipleLines = st.selectbox(\"Multiple Lines\", [\"Yes\", \"No\", \"No phone service\"])\n",
    "    InternetService = st.selectbox(\"Internet Service\", [\"DSL\", \"Fiber optic\", \"No\"])\n",
    "    OnlineSecurity = st.selectbox(\"Online Security\", [\"Yes\", \"No\", \"No internet service\"])\n",
    "    OnlineBackup = st.selectbox(\"Online Backup\", [\"Yes\", \"No\", \"No internet service\"])\n",
    "    DeviceProtection = st.selectbox(\"Device Protection\", [\"Yes\", \"No\", \"No internet service\"])\n",
    "    TechSupport = st.selectbox(\"Tech Support\", [\"Yes\", \"No\", \"No internet service\"])\n",
    "    StreamingTV = st.selectbox(\"Streaming TV\", [\"Yes\", \"No\", \"No internet service\"])\n",
    "    StreamingMovies = st.selectbox(\"Streaming Movies\", [\"Yes\", \"No\", \"No internet service\"])\n",
    "    Contract = st.selectbox(\"Contract\", [\"Month-to-month\", \"One year\", \"Two year\"])\n",
    "    PaperlessBilling = st.selectbox(\"Paperless Billing\", [\"Yes\", \"No\"])\n",
    "    PaymentMethod = st.selectbox(\n",
    "        \"Payment Method\",\n",
    "        [\"Electronic check\", \"Mailed check\", \"Bank transfer (automatic)\", \"Credit card (automatic)\"]\n",
    "    )\n",
    "    MonthlyCharges = st.number_input(\"Monthly Charges\", min_value=0.0, value=70.0)\n",
    "    TotalCharges = st.number_input(\"Total Charges\", min_value=0.0, value=1000.0)\n",
    "\n",
    "    data = {\n",
    "        \"gender\": gender,\n",
    "        \"SeniorCitizen\": SeniorCitizen,\n",
    "        \"Partner\": Partner,\n",
    "        \"Dependents\": Dependents,\n",
    "        \"tenure\": tenure,\n",
    "        \"PhoneService\": PhoneService,\n",
    "        \"MultipleLines\": MultipleLines,\n",
    "        \"InternetService\": InternetService,\n",
    "        \"OnlineSecurity\": OnlineSecurity,\n",
    "        \"OnlineBackup\": OnlineBackup,\n",
    "        \"DeviceProtection\": DeviceProtection,\n",
    "        \"TechSupport\": TechSupport,\n",
    "        \"StreamingTV\": StreamingTV,\n",
    "        \"StreamingMovies\": StreamingMovies,\n",
    "        \"Contract\": Contract,\n",
    "        \"PaperlessBilling\": PaperlessBilling,\n",
    "        \"PaymentMethod\": PaymentMethod,\n",
    "        \"MonthlyCharges\": MonthlyCharges,\n",
    "        \"TotalCharges\": TotalCharges\n",
    "    }\n",
    "\n",
    "    return pd.DataFrame([data])\n",
    "\n",
    "input_df = user_input()\n",
    "\n",
    "# ===============================\n",
    "# ENCODE INPUT (MUST MATCH TRAINING)\n",
    "# ===============================\n",
    "encoding_maps = {\n",
    "    \"gender\": {\"Female\": 0, \"Male\": 1},\n",
    "    \"Partner\": {\"No\": 0, \"Yes\": 1},\n",
    "    \"Dependents\": {\"No\": 0, \"Yes\": 1},\n",
    "    \"PhoneService\": {\"No\": 0, \"Yes\": 1},\n",
    "    \"PaperlessBilling\": {\"No\": 0, \"Yes\": 1},\n",
    "    \"MultipleLines\": {\"No phone service\": 0, \"No\": 1, \"Yes\": 2},\n",
    "    \"InternetService\": {\"DSL\": 0, \"Fiber optic\": 1, \"No\": 2},\n",
    "    \"OnlineSecurity\": {\"No\": 0, \"Yes\": 1, \"No internet service\": 2},\n",
    "    \"OnlineBackup\": {\"No\": 0, \"Yes\": 1, \"No internet service\": 2},\n",
    "    \"DeviceProtection\": {\"No\": 0, \"Yes\": 1, \"No internet service\": 2},\n",
    "    \"TechSupport\": {\"No\": 0, \"Yes\": 1, \"No internet service\": 2},\n",
    "    \"StreamingTV\": {\"No\": 0, \"Yes\": 1, \"No internet service\": 2},\n",
    "    \"StreamingMovies\": {\"No\": 0, \"Yes\": 1, \"No internet service\": 2},\n",
    "    \"Contract\": {\"Month-to-month\": 0, \"One year\": 1, \"Two year\": 2},\n",
    "    \"PaymentMethod\": {\n",
    "        \"Electronic check\": 0,\n",
    "        \"Mailed check\": 1,\n",
    "        \"Bank transfer (automatic)\": 2,\n",
    "        \"Credit card (automatic)\": 3\n",
    "    }\n",
    "}\n",
    "\n",
    "for col, mapping in encoding_maps.items():\n",
    "    input_df[col] = input_df[col].map(mapping)\n",
    "\n",
    "input_df = input_df[feature_columns]\n",
    "\n",
    "# ===============================\n",
    "# PREDICTION\n",
    "# ===============================\n",
    "if st.button(\"Predict Churn\"):\n",
    "    prediction = model.predict(input_df)[0]\n",
    "    probability = model.predict_proba(input_df)[0][1]\n",
    "\n",
    "    if prediction == 1:\n",
    "        st.error(f\"⚠️ Customer is likely to churn (Probability: {probability:.2%})\")\n",
    "    else:\n",
    "        st.success(f\"✅ Customer is unlikely to churn (Probability: {probability:.2%})\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a4c6bcb-8f3e-4c50-86bf-d17dd87a716c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
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
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
