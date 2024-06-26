import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.transforms.core import ParDo
from apache_beam.io.gcp.bigquery import WriteToBigQuery
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import json
import base64

# 定义数据预处理函数
class PreprocessData(beam.DoFn):
    def process(self, element):
        row = json.loads(element.decode('utf-8'))

        # One-hot 编码
        categorical_features = ['Gender', 'Occupation', 'BMI_Category']
        onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        onehot_encoded = onehot_encoder.fit_transform(np.array(row[categorical_features]).reshape(1, -1))
        onehot_features = onehot_encoder.categories_[0]
        for i, feature in enumerate(onehot_features):
            row[f'{feature}'] = int(onehot_encoded[0, i] == 1)

        # 数据标准化
        scaler = StandardScaler()
        numeric_features = ['Age', 'Sleep_Duration', 'Heart_Rate', 'Daily_Steps', 'Systolic', 'Diastolic']
        row[numeric_features] = scaler.fit_transform(np.array(row[numeric_features]).reshape(1, -1))[0]

        # 组合特征
        row['Sleep_Health_Index'] = row['Sleep_Duration'] * row['Quality_of_Sleep']
        row['Physical_Activity_Level_Stress_Level'] = row['Physical_Activity_Level'] * row['Stress_Level']

        # 返回处理后的完整字典
        return row

# 定义 Dataflow 管线
def run_pipeline(pipeline_options):
    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | 'ReadFromPubSub' >> ReadFromPubSub(subscription='projects/your-project-id/subscriptions/your-subscription-name')
            | 'PreprocessData' >> ParDo(PreprocessData())
            | 'WriteToBigQuery' >> WriteToBigQuery(
                table='your-project-id:your-dataset.preprocessed_sleep_data', 
                schema='your-bigquery-schema',
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

# 设置 Dataflow pipeline 选项
pipeline_options = PipelineOptions(
    runner='DataflowRunner',
    project='your-project-id',
    region='us-central1',
    temp_location='gs://your-bucket/temp',
    save_main_session=True,
)

# 运行数据预处理管道
run_pipeline(pipeline_options)
