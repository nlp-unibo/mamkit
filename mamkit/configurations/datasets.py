from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method
from typing import Type, List, Dict
from mamkit.components.datasets import (
    UKDebates,
    MMUSED,
    MMUSEDFallacy,
    MArg,
    InputMode
)

from pathlib import Path


class LoaderConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='task',
                   type_hint=str,
                   description='Task name.')
        config.add(name='input_mode',
                   type_hint=InputMode,
                   description='Task mode',
                   variants=[
                       InputMode.TEXT_ONLY,
                       InputMode.AUDIO_ONLY,
                       InputMode.TEXT_AUDIO
                   ])
        config.add(name='download_url',
                   type_hint=str,
                   description='URL where to download the dataset.')

        return config


class UKDebatesConfig(LoaderConfig):

    @classmethod
    @register_method(name='dataset',
                     tags={'data:ukdebates'},
                     namespace='mamkit',
                     component_class=UKDebates)
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.task = 'asd'
        config.get('task').allowed_range = lambda t: t in ['asd']
        config.download_url = 'http://argumentationmining.disi.unibo.it/dataset_aaai2016.tgz'

        config.add(name='folder_name',
                   value='UKDebates',
                   type_hint=str,
                   description='Dataset folder name containing data')
        config.add(name='audio_path',
                   value=Path('dataset', 'audio'),
                   type_hint=Path,
                   description='Relative path where to store audio content')

        return config


class MMUSEDConfig(LoaderConfig):

    @classmethod
    @register_method(name='dataset',
                     tags={'data:mmused'},
                     namespace='mamkit',
                     component_class=MMUSED)
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.get('task').variants = ['asd', 'acc']
        config.get('task').allowed_range = lambda p: p in ['asd', 'acc']
        config.download_url = 'https://zenodo.org/api/records/11179380/files-archive'
        config.add(name='folder_name',
                   value='MMUSED',
                   type_hint=str,
                   description='Dataset folder name containing data')
        config.add(name='deleted_ids',
                   value=['13_1988, 17_1992, 42_2016, 43_2016'],
                   type_hint=List[str],
                   description="List of USED dialogue IDs to be removed since no corresponding audio file can be found.")
        config.add(name='files_filename',
                   value='files',
                   type_hint=str,
                   description='Folder name where to store downloaded files')
        config.add(name='aeneas_filename',
                   value='aenaes',
                   type_hint=str,
                   description='Folder name where to store AENEAS files')
        config.add(name='audio_filename',
                   value='debates_audio_recordings',
                   type_hint=str,
                   description='Folder name where to store audio files')
        config.add(name='datasets_filename',
                   value='datasets',
                   type_hint=str,
                   description='Folder name where to store dataset files')
        config.add(name='transcripts_filename',
                   value='transcripts',
                   type_hint=str,
                   description='Folder name where to store transcript files')
        config.add(name='alignment_filename',
                   value='alignment_results',
                   type_hint=str,
                   description='Folder name where to store audio alignment files')
        config.add(name='audio_clips_filename',
                   value='audio_clips',
                   type_hint=str,
                   description='Folder name where to store audio clips files')
        config.add(name='processed_filename',
                   value='MM-USElecDeb60to16',
                   type_hint=str,
                   description='Folder name where to store the final dataset')

        return config


class MMUSEDFallacyConfig(LoaderConfig):

    @classmethod
    @register_method(name='dataset',
                     tags={'data:mmused-fallacy'},
                     namespace='mamkit',
                     component_class=MMUSEDFallacy)
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.task = 'afc'
        config.get('task').allowed_range = lambda p: p in ['afc']
        config.download_url = 'https://zenodo.org/api/records/11179390/files-archive'
        config.add(name='sample_rate',
                   value=16000,
                   type_hint=int,
                   is_required=True,
                   description='Frequency to which audio is (re-)sampled.')
        config.add(name='folder_name',
                   value='MMUSED-fallacy',
                   type_hint=str,
                   is_required=True,
                   description='Dataset folder name containing data')
        config.add(name='resource_filename',
                   value='files',
                   type_hint=str,
                   description='Folder name where to store downloaded files',
                   is_required=True)
        config.add(name='audio_filename',
                   value='debates_audio_recordings',
                   type_hint=str,
                   description='Folder name where to store audio files',
                   is_required=True)
        config.add(name='audio_clips_filename',
                   value='audio_clips',
                   type_hint=str,
                   description='Folder name where to store audio clips files',
                   is_required=True)

        return config


class MArgConfig(LoaderConfig):

    @classmethod
    @register_method(name='dataset',
                     tags={'data:marg'},
                     namespace='mamkit',
                     component_class=MArg)
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.task = 'arc'
        config.get('task').allowed_range = lambda p: p in ['arc']
        config.download_url = 'https://zenodo.org/api/records/5653504/files-archive'
        config.add(name='confidence',
                   type_hint=float,
                   variants=[
                       0.85,
                       1.00
                   ],
                   is_required=True,
                   description='Annotation confidence to filter out samples.')
        config.add(name='folder_name',
                   value='MArg',
                   type_hint=str,
                   is_required=True,
                   description='Dataset folder name containing data')
        config.add(name='feature_path',
                   value=Path('data',
                              'preprocessed full dataset',
                              'full_feature_extraction_dataset.csv'),
                   type_hint=Path,
                   description='Relative path where feature dataset is stored',
                   is_required=True)
        config.add(name='aggregated_path',
                   value=Path('annotated dataset', 'aggregated_dataset.csv'),
                   type_hint=Path,
                   description='Relative path where aggregated dataset is stored',
                   is_required=True)
        config.add(name='audio_path',
                   value=Path('data', 'audio sentences'),
                   type_hint=Path,
                   description='Relative path where audio files are stored',
                   is_required=True)
        config.add(name='speakers_map',
                   value={
                       'Chris Wallace': 'Chris Wallace',
                       'Vice President Joe Biden': 'Joe Biden',
                       'President Donald J. Trump': 'Donald Trump',
                       'Chris Wallace:': 'Chris Wallace',
                       'Kristen Welker': 'Kristen Welker',
                       'Donald Trump': 'Donald Trump',
                       'Joe Biden': 'Joe Biden',
                       'George Stephanopoulos': 'George Stephanopoulos',
                       'Nicholas Fed': 'Audience Member 1',
                       'Kelly Lee': 'Audience Member 2',
                       'Anthony Archer': 'Audience Member 3',
                       'Voice Over': 'Voice Over',
                       'Cedric Humphrey': 'Audience Member 4',
                       'George Stephanopoulus': 'George Stephanopoulos',
                       'Angelia Politarhos': 'Audience Member 5',
                       'Speaker 1': 'Voice Over',
                       'Nathan Osburn': 'Audience Member 6',
                       'Andrew Lewis': 'Audience Member 7',
                       'Speaker 2': 'Voice Over',
                       'Michele Ellison': 'Audience Member 8',
                       'Mark Hoffman': 'Audience Member 9',
                       'Mieke Haeck': 'Audience Member 10',
                       'Speaker 3': 'Voice Over',
                       'Keenan Wilson': 'Audience Member 11',
                       'Savannah Guthrie': 'Savannah Guthrie',
                       'President Trump': 'Donald Trump',
                       'Jacqueline Lugo': 'Audience Member 12',
                       'Barbara Peña': 'Audience Member 13',
                       'Isabella Peña': 'Audience Member 14',
                       'Savannah': 'Savannah Guthrie',
                       'Cristy Montesinos Alonso': 'Audience Member 15',
                       'Adam Schucher': 'Audience Member 16',
                       'Moriah Geene': 'Audience Member 17',
                       'Cindy Velez': 'Audience Member 18',
                       'Paulette Dale': 'Audience Member 19',
                       'Susan Page': 'Susan Page',
                       'Kamala Harris': 'Kamala Harris',
                       'Mike Pence': 'Mike Pence',
                       'Kamala Harris ': 'Kamala Harris'
                   },
                   type_hint=Dict[str, str],
                   description='File to file speaker mapping',
                   is_required=True)
        config.add(name='debate_id_map',
                   value={
                       'us_election_2020_1st_presidential_debate_part1_timestamp.csv': '00',
                       'us_election_2020_1st_presidential_debate_part2_timestamp.csv': '01',
                       'us_election_2020_2nd_presidential_debate_part1_timestamp.csv': '02',
                       'us_election_2020_2nd_presidential_debate_part2_timestamp.csv': '03',
                       'us_election_2020_biden_town_hall_part1_timestamp.csv': '04',
                       'us_election_2020_biden_town_hall_part2_timestamp.csv': '05',
                       'us_election_2020_biden_town_hall_part3_timestamp.csv': '06',
                       'us_election_2020_biden_town_hall_part4_timestamp.csv': '07',
                       'us_election_2020_biden_town_hall_part5_timestamp.csv': '08',
                       'us_election_2020_biden_town_hall_part6_timestamp.csv': '09',
                       'us_election_2020_biden_town_hall_part7_timestamp.csv': '10',
                       'us_election_2020_trump_town_hall_1_timestamp.csv': '11',
                       'us_election_2020_trump_town_hall_2_timestamp.csv': '12',
                       'us_election_2020_trump_town_hall_3_timestamp.csv': '13',
                       'us_election_2020_trump_town_hall_4_timestamp.csv': '14',
                       'us_election_2020_vice_presidential_debate_1_timestamp.csv': '15',
                       'us_election_2020_vice_presidential_debate_2_timestamp.csv': '16'
                   },
                   type_hint=Dict[str, str],
                   description='File to id debate mapping',
                   is_required=True)
        config.add(name='file_map_timestamp',
                   value={
                       'us_election_2020_1st_presidential_debate_part1_timestamp.csv': 'us_election_2020_1st_presidential_debate_split.csv',
                       'us_election_2020_1st_presidential_debate_part2_timestamp.csv': 'us_election_2020_1st_presidential_debate_split.csv',
                       'us_election_2020_2nd_presidential_debate_part1_timestamp.csv': 'us_election_2020_2nd_presidential_debate_split.csv',
                       'us_election_2020_2nd_presidential_debate_part2_timestamp.csv': 'us_election_2020_2nd_presidential_debate_split.csv',
                       'us_election_2020_biden_town_hall_part1_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
                       'us_election_2020_biden_town_hall_part2_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
                       'us_election_2020_biden_town_hall_part3_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
                       'us_election_2020_biden_town_hall_part4_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
                       'us_election_2020_biden_town_hall_part5_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
                       'us_election_2020_biden_town_hall_part6_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
                       'us_election_2020_biden_town_hall_part7_timestamp.csv': 'us_election_2020_biden_town_hall_split.csv',
                       'us_election_2020_trump_town_hall_1_timestamp.csv': 'us_election_2020_trump_town_hall_split.csv',
                       'us_election_2020_trump_town_hall_2_timestamp.csv': 'us_election_2020_trump_town_hall_split.csv',
                       'us_election_2020_trump_town_hall_3_timestamp.csv': 'us_election_2020_trump_town_hall_split.csv',
                       'us_election_2020_trump_town_hall_4_timestamp.csv': 'us_election_2020_trump_town_hall_split.csv',
                       'us_election_2020_vice_presidential_debate_1_timestamp.csv': 'us_election_2020_vice_presidential_debate_split.csv',
                       'us_election_2020_vice_presidential_debate_2_timestamp.csv': 'us_election_2020_vice_presidential_debate_split.csv'},
                   type_hint=Dict[str, str],
                   description='Debate file to split file mapping',
                   is_required=True)

        return config
