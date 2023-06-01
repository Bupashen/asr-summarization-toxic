import os
import time
import json
import umap
import copy
import shutil
import whisper
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Union
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from components.speaker_model import SpeakerModel
from components.voice_activity_detector import VoiceActivityDetector
from utils.utils_engine import (dump_to_json, 
                                prepare_audio,
                                break_down_audio,
                                get_duration,
                                TimeInterval)
from deeppavlov import build_model
import re
from yargy import Parser, or_
from yargy.pipelines import morph_pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import MBartTokenizer, MBartForConditionalGeneration


class ASREngine:
    def __init__(
        self, 
        path_to_whisper: str,
        path_to_speaker_model: str,
        path_to_vad: str,
        path_to_data_storage: str,
        path_to_speech_detector: str = None,
        device: str = 'cpu',
        sample_rate: int = 16_000,
        clustering_method: str = 'AgCl',
        drop_empty: bool = True
    ):
        self.sample_rate = sample_rate
        self._whisper_model = whisper.load_model(path_to_whisper, device=device)
        if path_to_speech_detector is not None:
            self._speech_detector = whisper.load_model(
                path_to_speech_detector, device=device
            )
        else:
            self._speech_detector = self._whisper_model
        self._speaker_model = SpeakerModel(path_to_speaker_model, sample_rate, device)
        self._vad = VoiceActivityDetector(path_to_vad, sample_rate, device)
        self.clustering_method = clustering_method
        self.ner_model = build_model('dp_ner_config.json')
        self._prepare_summary_parser()
        self._prepare_naming_condition()
        self.drop_empty = drop_empty
        self.path_to_data_storage = path_to_data_storage
        self.tox_tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny-toxicity')
        self.tox_detector = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny-toxicity')
        self.sum_tokenizer = MBartTokenizer.from_pretrained("Kirili4ik/mbart_ruDialogSum")
        self.sum_model = MBartForConditionalGeneration.from_pretrained("Kirili4ik/mbart_ruDialogSum")
        if torch.cuda.is_available():
            self.tox_detector.cuda()
        self._set_data_storage()
        print('---Model has been initialized---', flush=True)

    def _prepare_naming_condition(self):
        naming_condition_pattern = [
            "меня зовут",
            "добрый день, меня зовут",
            "всем привет, меня зовут",
            "позвольте представиться",
            "да, всем привет, я",
            "зовут меня",
            "представлюсь, меня зовут",
            "давайте представлюсь, меня звать",
            "позвольте представлюсь",
            "давайте знакомиться, я",
            "давайте знакомиться, меня зовут",
        ]
        self.NAMING_CONDITION = Parser(morph_pipeline(naming_condition_pattern))

    def _prepare_summary_parser(self):
        pattern_inceptum = [
            'а теперь к итогам', 
            'а теперь к выводам'
        ]
        pattern_medium = [
            'во-первых',
            'во-вторых',
            'в-третьих',
            'в-четвертых',
            'в-пятых',
            'в-шестых',
            'в-седьмых',
            'в-восьмых',
            'в-девятых',
            'в-десятых',
        ]
        pattern_factus = [
            'спасибо за внимание',
            'готов ответить на вопросы'
        ]

        self.INCEPTUM = Parser(morph_pipeline(pattern_inceptum))
        self.COMPREHENSION = [
            Parser(or_(morph_pipeline([item]), morph_pipeline([re.sub('-', ' ', item)])))
            for item in pattern_medium
        ]
        self.FACTUS = Parser(morph_pipeline(pattern_factus))

    def _create_dir(self, path_to_dir: str) -> None:
        if os.path.exists(path_to_dir):
            shutil.rmtree(path_to_dir)
        os.mkdir(path_to_dir)

    def _set_data_storage(self):
        assert os.path.exists(self.path_to_data_storage)
        self._input_data = os.path.join(self.path_to_data_storage, 'input_data')
        self._create_dir(self._input_data)
        self._intermidiate_data = os.path.join(self.path_to_data_storage, 'intermidiate_data')
        self._create_dir(self._intermidiate_data)
        self._output_data = os.path.join(self.path_to_data_storage, 'output_data')
        self._create_dir(self._output_data)
        self._vad_segments_dir = os.path.join(self._intermidiate_data, 'vad_segments')
        self._create_dir(self._vad_segments_dir)

        self._path_to_audio_prepared = os.path.join( self._input_data, 'audio_prepared.wav')
        self._path_to_transcription = os.path.join( self._intermidiate_data, 'transcription.json')
        self._path_to_vad_result = os.path.join( self._intermidiate_data, 'vad_result.json')
        self._path_to_vad_segments = os.path.join( self._intermidiate_data, 'vad_segments.json')
        self._path_to_speech_flags = os.path.join( self._intermidiate_data, 'speech_flags.json')
        self._path_to_speech_embeddings = os.path.join( self._intermidiate_data, 'speech_embeddings.npy')
        self._path_to_segments_labels = os.path.join( self._intermidiate_data, 'segments_labels.json')
        self._path_to_result = os.path.join( self._output_data, 'result.json')

    def transcribe(
        self, path_to_file: str, 
        transcription_title: str, 
        n_speakers: int,
        use_naming: bool = True,
        seek_summary: bool = True,
        find_toxic: bool = True,
        find_whole_summary: bool = True
    ) -> None:
        transcription_start = time.time()
        print('---Preparing audio---')
        try:
            prepare_audio(
                path_to_file, 
                self._input_data, 
                self.sample_rate
            )
        except Exception as e:
            print('Can\'t prepare audio for transcription')
            raise e
        end = time.time()
        get_duration(transcription_start, end)

        print('---Transcribing audio---')
        try:
            transcription = self._whisper_model.transcribe(
                self._path_to_audio_prepared, language='ru', verbose=False
            )
            dump_to_json(transcription, self._path_to_transcription)
            whisper_timestamps = [
                {'start': item['start'], 'end': item['end']} 
                for item in transcription['segments']
            ]
        except Exception as e:
            print('Can\'t transcribe audio')
            raise e
        
        start = time.time()
        print('---Performing VAD---')
        try:
            vad_result = self._get_vad_result(self._path_to_audio_prepared)
            dump_to_json(vad_result, self._path_to_vad_result)
            vad_segments = self._extend_vad_result_with_whisper(
                whisper_timestamps, vad_result
            )
            dump_to_json(vad_segments, self._path_to_vad_segments)
        except Exception as e:
            print('Can\'t perform VAD on audio')
            raise e
        end = time.time()
        get_duration(start, end)

        break_down_audio(
            self._path_to_audio_prepared,
            vad_segments,
            self._vad_segments_dir
        )

        print('---Non speech segments inspection---')
        try:
            is_speech_flags = self._inspect_for_non_speech(vad_segments)
            for item, flag in zip(vad_segments, is_speech_flags):
                item['is_speech'] = flag
            dump_to_json(is_speech_flags, self._path_to_speech_flags)
        except Exception as e:
            print('Can\'t perform non speech inspection')
            raise e

        print('---Getting speaker embeddings---')
        try:
            embeddings = self._get_embeddings(len(vad_segments))
            np.save(self._path_to_speech_embeddings, embeddings)
        except Exception as e:
            print('Can\'t get speaker embeddings')
            raise e
        
        start = time.time()
        print('---Clustering segments---')
        try:
            segments_labels = self._clusterize(embeddings, n_speakers)
            for item, label in zip(vad_segments, segments_labels):
                item['speaker_label'] = label
            dump_to_json(segments_labels, self._path_to_segments_labels)
        except Exception as e:
            print('Can\'t clusterize segment')
            raise e
        end = time.time()
        get_duration(start, end)

        start = time.time()
        print('---Matching segments with transcriptions---')
        try:
            transcription_to_vad = self._match_whisper_with_vad(
                vad_segments, transcription
            )
            for text, segment in zip(transcription_to_vad, vad_segments):
                segment['text'] = text
        except Exception as e:
            print('Can\'t match segments with transcriptions')
            raise e
        end = time.time()
        get_duration(start, end)

        start = time.time()
        print('---Postprocessing result---')
        try:   
            if self.clustering_method == 'DBSCAN':
                n_speakers = max(set(segments_labels)) + 1
            vad_segments_clamped = self._clamp_segments(vad_segments)
            if self.drop_empty:
                vad_segments_final = self._drop_empty(vad_segments_clamped)
                vad_segments_final = self._clamp_segments(vad_segments_final)
            else:
                vad_segments_final = vad_segments_clamped
            vad_segments_post = self._postprocess_segments(vad_segments_final)
            if use_naming:
                label_to_name = self._get_speaker_names(vad_segments_post, n_speakers)
            else:
                label_to_name = [None]*n_speakers
            if seek_summary:
                summary = self._get_summary(transcription, vad_segments_post)
            else:
                summary = {
                    'summary': [],
                    'bolding_bubbles': {}
                }
            if find_toxic: 
                toxicity_markers = self._find_toxic_utterances(vad_segments_post)
            else:
                toxicity_markers = [None]*len(vad_segments_post)
            if find_whole_summary:
                whole_summary = self._get_whole_summary(transcription['text'])
            else:
                whole_summary = []
        except Exception as e:
            print('Can\t postprocess result')
            raise e
        end = time.time()
        get_duration(start, end)

        print('---Saving results---')
        try:
            result = self._construct_json(
                vad_segments_post, 
                path_to_file, 
                transcription_title, 
                label_to_name, 
                summary, 
                toxicity_markers,
                whole_summary
            )
            with open(self._path_to_result, 'w') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        except:
            raise Exception('Can\'t save results')  
        transcription_end = time.time()
        print('Overall time')
        get_duration(transcription_start, transcription_end) 


    """VAD"""
    def _get_vad_result(self, path_to_audio: str) -> List[Dict]:
        vad_result = self._vad.get_timestamps_in_seconds(
            path_to_audio
        )
        vad_result_prolonged = self._prolong_vad_segments(vad_result)
        return vad_result_prolonged

    def _prolong_vad_segments(self, vad_results: List[Dict]) -> List[Dict]:
        vad_results_n = []
        PROLONGING_FACTOR = 0.250
        for item in vad_results:
            duration = item['end'] - item['start']
            if duration < PROLONGING_FACTOR:
                delta = (PROLONGING_FACTOR - duration) / 2
                vad_results_n.append({
                    'start': max(item['start'] - delta, 0),
                    'end': item['end'] + delta
                })
            else:
                vad_results_n.append({**item})
        return vad_results_n

    def _extend_vad_result_with_whisper(
            self, whisper_timestamps: List[Dict], vad_result: List[Dict] 
        ) -> List[Dict]:
        whisper_to_vad_map = {}
        for i, whisper_seg in enumerate(whisper_timestamps):
            overlapped_vads = 0
            w_seg = TimeInterval(start=whisper_seg['start'], end=whisper_seg['end'])
            for vad_seg in vad_result:
                v_seg = TimeInterval(start=vad_seg['start'], end=vad_seg['end'])
                if w_seg.intersects(v_seg):
                    overlapped_vads += 1
            whisper_to_vad_map[i] = overlapped_vads
        free_whisper_segs = [whisper_timestamps[i] 
                                for i, overlapped_vads in whisper_to_vad_map.items()
                                if overlapped_vads == 0]
        vad_result_n = []
        for item in vad_result:
            item_n = {**item}
            item_n['source'] = 'vad'
            vad_result_n.append(item_n)
        free_whisper_segs_n = []
        for item in free_whisper_segs:
            item_n = {**item}
            item_n['source'] = 'whisper'
            free_whisper_segs_n.append(item_n)
        vad_segments = self._fuse_lists_by_field(
            free_whisper_segs_n, vad_result_n, 'start'
        )
        return vad_segments

    def _fuse_lists_by_field(
            self, L_1: List[Dict], L_2: List[Dict], field: str
        ) -> List[Dict]:
        L_fused = []
        i_1 = 0
        i_2 = 0
        while i_1 < len(L_1) and i_2 < len(L_2):
            if L_1[i_1][field] <= L_2[i_2][field]:
                L_fused.append(L_1[i_1])
                i_1 += 1
            else:
                L_fused.append(L_2[i_2])
                i_2 += 1
        if i_1 == len(L_1):
            L_fused += L_2[i_2:]
        elif i_2 == len(L_2):
            L_fused += L_1[i_1:]  
        return L_fused
    
    """SPEECH DETECTION"""
    def _inspect_for_non_speech(self, vad_segments: List[Dict]) -> List[bool]:
        is_speech_flags = []
        for i, item in tqdm(enumerate(vad_segments)):
            if item['source'] == 'whisper':
                is_speech_flags.append(True)
            else:
                segment_filepath = os.path.join(self._vad_segments_dir, f'{i}.wav')
                tr = self._speech_detector.transcribe(segment_filepath)
                if tr['language'] == 'ru':
                    is_speech_flags.append(True)
                else:
                    is_speech_flags.append(False)
        return is_speech_flags
    
    """SPEAKER MODELING"""
    def _get_embeddings(self, n_segments: int):
        embeddings = []
        for i in tqdm(range(n_segments)):
            segment_filepath = os.path.join(self._vad_segments_dir, f'{i}.wav')
            try:
                segment_embedding = self._speaker_model.get_embedding(segment_filepath)
            except:
                segment_embedding = np.zeros((1, 192))
            embeddings.append(segment_embedding)
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
    
    """CLUSTERING"""
    def _clusterize(
            self, embeddings: np.ndarray, n_speakers: int
        ) -> List[int]:
        reducer = umap.UMAP()
        embeddings_r = reducer.fit_transform(embeddings)
        if self.clustering_method == 'AgCl':    
            clusterer = AgglomerativeClustering(n_speakers)
        elif self.clustering_method == 'DBSCAN':
            clusterer = DBSCAN()
        clusterer.fit(embeddings_r)
        if self.clustering_method == 'AgCl':    
            labels = clusterer.labels_.tolist()
        elif self.clustering_method == 'DBSCAN':
            labels = [i+1 for i in clusterer.labels_.tolist()]
        return labels
    
    """TRANSCRIPTION MATCHING"""
    def _cross_associate_segments(
            self, vad_segments: List[Dict], transcription: List[Dict]
        ) -> Dict:
        cross_association_table = {}
        for whisper_segment in transcription['segments']:
            w_seg = TimeInterval(
                start=whisper_segment['start'], end=whisper_segment['end']
            )
            vad_segments_ids = []
            for i in range(len(vad_segments)):
                v_seg = TimeInterval(
                    start=vad_segments[i]['start'], end=vad_segments[i]['end']
                )
                if w_seg.intersects(v_seg):
                    vad_segments_ids.append(i)
            cross_association_table[whisper_segment['id']] = vad_segments_ids
        return cross_association_table

    def _match_transcription(
            self,
            vad_segments: List[Dict], 
            transcription: List[Dict],
            cross_assossiation_table: List[Dict]
        ) -> List[List[str]]:
        transcription_to_vad = [None]*len(vad_segments)
        v_segs = [TimeInterval(item['start'], item['end'])
                for item in vad_segments]
        for whisper_id, vad_segments_ids in cross_assossiation_table.items():
            w_seg = TimeInterval(
                start=transcription['segments'][whisper_id]['start'],
                end=transcription['segments'][whisper_id]['end']
            )
            longest_speech_seg = {
                'id': None, 'intersection': 0
            }
            longest_non_speech_seg = {
                'id': None, 'intersection': 0
            }
            for vad_id in vad_segments_ids:
                cur_intersection = w_seg._get_intersection(v_segs[vad_id])
                if (vad_segments[vad_id]['is_speech']
                    and cur_intersection > longest_speech_seg['intersection']):
                    longest_speech_seg['id'] = vad_id
                    longest_speech_seg['intersection'] = cur_intersection
                elif (not vad_segments[vad_id]['is_speech']
                    and cur_intersection > longest_non_speech_seg['intersection']):
                    longest_non_speech_seg['id'] = vad_id
                    longest_non_speech_seg['intersection'] = cur_intersection
            if longest_speech_seg['id'] is not None:
                idx = longest_speech_seg['id']
            elif longest_non_speech_seg['id'] is not None:
                idx = longest_non_speech_seg['id']
            else:
                continue
            if transcription_to_vad[idx] is not None:
                transcription_to_vad[idx].append(
                    transcription['segments'][whisper_id]['text']
                )
            else:
                transcription_to_vad[idx] = [
                    transcription['segments'][whisper_id]['text']
                ]
        return transcription_to_vad

    def _match_whisper_with_vad(
            self, vad_segments: List[Dict], transcription: List[Dict]
        ) -> Dict:
        cross_association_table = self._cross_associate_segments(
            vad_segments, transcription
        )
        transcription_to_vad = self._match_transcription(
            vad_segments, transcription, cross_association_table
        )
        return transcription_to_vad
    
    """POSTPROCESSING"""
    def _clamp_segments(
            self, vad_segments: List[Dict], clamping_eps: float = 3
        ) -> List[Dict]:
        vad_segments_clamped = []
        cur_cluster = {
            'start': vad_segments[0]['start'],
            'end': vad_segments[0]['end'],
            'speaker_label': vad_segments[0]['speaker_label'],
            'text': []
        }
        for i, item in enumerate(vad_segments):
            if (item['speaker_label'] == cur_cluster['speaker_label']
                and cur_cluster['end'] + clamping_eps >= item['start']):
                cur_cluster['end'] = item['end']
                if item['text'] is not None:
                    cur_cluster['text'] += item['text']
            else:
                vad_segments_clamped.append(cur_cluster)
                cur_cluster = {
                    'start': item['start'],
                    'end': item['end'],
                    'speaker_label': item['speaker_label'],
                    'text': copy.deepcopy(item['text']) if item['text'] is not None else []
                }
            if i == len(vad_segments) - 1:
                vad_segments_clamped.append(cur_cluster)
        return vad_segments_clamped

    def _drop_empty(self, vad_segments_clamped: List[Dict]) -> List[Dict]:
        vad_segments_clamped_n = []
        for item in vad_segments_clamped:
            if len(item['text']) > 0:
                vad_segments_clamped_n.append(item)
        return vad_segments_clamped_n

    def _postprocess_segments(self, vad_segments: List[Dict]) -> List[Dict]:
        vad_segments_post = []
        for item in vad_segments:
            item_n = copy.deepcopy(item)
            item_n['text'] = (''.join(item['text'])).strip()
            vad_segments_post.append(item_n)
        return vad_segments_post
    
    def _get_speaker_names(self, vad_segments_post: List[Dict], n_speakers: int) -> List[str]:
        utterances = [item['text'] for item in vad_segments_post]
        naming_scopes = []
        for item in utterances:
            name_conditions = list(self.NAMING_CONDITION.findall(item))
            if len(name_conditions) > 0:
                naming_scope = min(name_conditions[0].span.stop, int(os.environ['NAMING_SCOPE']))
            else:
                naming_scope = int(os.environ['NAMING_SCOPE'])
            naming_scopes.append(naming_scope)
        utterances_cut = []
        for naming_scope, utterance in zip(naming_scopes, utterances):
            area_of_interest = utterance[naming_scope:int(os.environ['NAMING_SCOPE'])+1]
            if len(area_of_interest) > 0:
                utterances_cut.append(area_of_interest)
            else:
                utterances_cut.append("---")
        speaker_labels = [item['speaker_label'] for item in vad_segments_post]

        ner_results = self.ner_model(utterances_cut)
        fio_list = self._get_first_per_chains(ner_results)

        label_to_name = self._match_speaker_with_name(
            speaker_labels, fio_list, n_speakers
        )
        return label_to_name

    def _get_first_per_chains(self, ner_results: List) -> List:
        per_chains = []
        texts_tok, tags_tok = ner_results
        for tokens, tags in zip(texts_tok, tags_tok):
            cur_names = []
            found_chain = False
            for token, tag in zip(tokens, tags):
                if tag not in ['B-PER', 'I-PER'] and not found_chain:
                    continue
                elif tag in ['B-PER', 'I-PER'] and not found_chain:
                    cur_names.append(token)
                    found_chain = True
                elif tag in ['B-PER', 'I-PER'] and found_chain:
                    cur_names.append(token)
                elif tag not in ['B-PER', 'I-PER'] and found_chain:
                    break
            per_chains.append(cur_names)
        per_chains = [' '.join(item) if len(item) > 1 else '' for item in per_chains]
        return per_chains
    
    def _match_speaker_with_name(
        self,
        speaker_labels: List[int], 
        fio_list: List[str], 
        n_speakers: int
    ) -> List[str]:
        available_labels = set(range(n_speakers))
        label_to_name = [None]*n_speakers
        for label, fio in zip(speaker_labels, fio_list):
            if (available_labels is not None
                and label in available_labels
                and fio != ''):
                label_to_name[label] = fio
                available_labels.remove(label)
        return label_to_name
    
    def _get_summary_entries(self, text: str) -> Tuple[Union[List[str], List[Tuple[int]]]]:

        results = []
        absolute_spans = []

        inceptum = list(self.INCEPTUM.findall(text))
        inceptum_span = None
        if len(inceptum) > 0:
            inceptum = inceptum[-1]
            inceptum_span = inceptum.span
        
        factus = list(self.FACTUS.findall(text))
        factus_span = None
        if len(factus) > 0:
            factus = factus[0]
            factus_span = factus.span

        if ((inceptum_span is not None and factus_span is not None)
            and (inceptum.span.stop < factus.span.start)):

            area_of_interest = text[inceptum.span.stop : factus.span.start]
            span_shift = inceptum.span.stop
            comprehension_spans = []

            for comp in self.COMPREHENSION:
                current_match = list(comp.findall(area_of_interest))
                if len(current_match) > 0:
                    current_match = current_match[0].span
                    comprehension_spans.append(current_match)

            for i in range(len(comprehension_spans)):
                if i < len(comprehension_spans) - 1:
                    cur_slice = slice(comprehension_spans[i].stop, comprehension_spans[i+1].start)
                    cur_span = (span_shift+comprehension_spans[i].stop, span_shift+comprehension_spans[i+1].start)
                    cur_res = area_of_interest[cur_slice]
                else:
                    cur_span = (span_shift+comprehension_spans[i].stop, span_shift+len(area_of_interest))
                    cur_res = area_of_interest[comprehension_spans[i].stop:]
                
                if len(cur_res) > 0:
                    first_letter_match = re.search('[A-Za-z]|[А-Яа-я]', cur_res)
                    first_letter_pos = first_letter_match.start()
                    cur_span = (cur_span[0]+first_letter_pos, cur_span[1])
                    cur_res = cur_res[first_letter_pos:].strip().capitalize()

                results.append(cur_res)
                absolute_spans.append(cur_span)

        return results, absolute_spans

    def _join_segments(self, vad_segments_post: List[Dict]) -> Tuple:
        whole_text = []
        bubble_spans = []
        total_length = 0
        for item in vad_segments_post:
            bubble = item['text'].strip()
            bubble_spans.append((total_length, total_length+len(bubble)))
            total_length += len(bubble) + 1
            whole_text.append(bubble)
        whole_text = ' '.join(whole_text)
        return whole_text, bubble_spans

    def _get_summary_partition(self, absolute_spans_tech: List[Tuple[int]], bubble_spans: List[Tuple[int]]) -> List[Dict]:
        summary_partition = []
        for abs_span in absolute_spans_tech:
            abs_span_interval = TimeInterval(*abs_span)
            cur_abs_span_partition = {}
            for i, bub_span in enumerate(bubble_spans):
                if bub_span[0] >= abs_span[1]:
                    break
                bub_span_interval = TimeInterval(*bub_span)
                if abs_span_interval.intersects(bub_span_interval):
                    cur_bub_partition_start = max(abs_span[0], bub_span[0])
                    cur_bub_partition_end = min(abs_span[1], bub_span[1])
                    cur_abs_span_partition[i] = (
                        cur_bub_partition_start - bub_span[0], 
                        cur_bub_partition_end - bub_span[0]
                    )
            summary_partition.append(cur_abs_span_partition)
        return summary_partition


    def _get_html_bolding_bubbles(self, summary_partition: List[Dict]) -> Dict:
        bubble_idx_to_partitions = {}
        for summary in summary_partition:
            for bubble_idx, partition in summary.items():
                if str(bubble_idx) in bubble_idx_to_partitions:
                    bubble_idx_to_partitions[str(bubble_idx)].append(partition)
                else:
                    bubble_idx_to_partitions[str(bubble_idx)] = [partition]
        return bubble_idx_to_partitions


    def _get_summary(self, transcription: Dict, vad_segments_post: List[Dict]) -> List[Dict]:
        summary, _ = self._get_summary_entries(transcription['text'])
        if len(summary) == 0:
            summary = {
                'summary': [],
                'bolding_bubbles': {}
            }
        else:
            whole_text, bubble_spans = self._join_segments(vad_segments_post)
            _, absolute_spans_tech = self._get_summary_entries(whole_text)
            if len(absolute_spans_tech) != len(summary):
                summary = [
                    {'text': item, 'partitions': {}} 
                    for item in summary
                ]
                summary = {
                    'summary': summary,
                    'bolding_bubbles': {}
                }
            else:
                summary_partition = self._get_summary_partition(absolute_spans_tech, bubble_spans)
                html_bolding_bubbles = self._get_html_bolding_bubbles(summary_partition)
                summary = [
                    {'text': item, 'partitions': partition} 
                    for item, partition in zip(summary, summary_partition)
                ]
                summary = {
                    'summary': summary,
                    'bolding_bubbles': html_bolding_bubbles
                }
        return summary
    
    def _find_toxic_utterances(self, vad_segments_post: List[Dict]) -> List[bool]:
        utterances = [item['text'] for item in vad_segments_post]
        splitted_texts = []
        markers = []
        for i, item in enumerate(utterances):
            cur_text_splitted = self._split_text(item)
            markers += [i]*len(cur_text_splitted)
            splitted_texts += cur_text_splitted
        toxicities = self._text2toxicity(splitted_texts)
        toxicity_markers = self._get_toxicity_markers(toxicities, markers)
        return toxicity_markers
    
    def _split_text(self, text: str, chunk_size: int = None) -> List[str]:
        text_enc = self.tox_tokenizer(text)['input_ids']
        if chunk_size is None:
            chunk_size = self.tox_tokenizer.model_max_length
        n_chunks = int(len(text_enc) // chunk_size if len(text_enc) % chunk_size == 0 
                    else len(text_enc) // chunk_size + 1)
        chunks = []
        for i in range(n_chunks):
            chunks.append(
                self.tox_tokenizer.decode(text_enc[i*chunk_size : (i+1)*chunk_size], skip_special_tokens=True)
            )
        return chunks
    
    def _text2toxicity(self, text, aggregate: bool = True, theshold: float = 0.8):
        with torch.no_grad():
            inputs = self.tox_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(self.tox_detector.device)
            proba = torch.sigmoid(self.tox_detector(**inputs).logits).cpu().numpy()
        if isinstance(text, str):
            proba = proba[0]
        if aggregate:
            return ((1 - proba.T[0] * (1 - proba.T[-1])) > theshold).tolist()
        return proba
    
    def _get_toxicity_markers(self, toxicities: List[bool], markers: List[int]) -> List[bool]:
        ids_set = set(markers)
        toxicity_markers = [None] * len(ids_set)
        toxicities = np.array(toxicities)
        markers = np.array(markers)
        for item in ids_set:
            toxicity_markers[item] = any(toxicities[markers == item])
        return toxicity_markers
    
    def _get_whole_summary(self, transcription_text: str) -> str:
        chunks = self._split_text(transcription_text, self.sum_tokenizer.model_max_length)
        input_ids = self.sum_tokenizer(
            chunks,
            max_length=600,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        output_ids = self.sum_model.generate(
            input_ids=input_ids,
            top_k=0,
            num_beams=3,
            no_repeat_ngram_size=3
        )

        whole_summary = []
        for item in output_ids:
            sent_decoded = self.sum_tokenizer.decode(item, skip_special_tokens=True)
            whole_summary.append(sent_decoded)
        whole_summary = '. '.join(whole_summary)
        return whole_summary

    """SAVING RESULT"""
    def _construct_json(
        self,
        resulted_track: List[Dict],
        path_to_file: str,
        transcription_title: str,
        label_to_name: List[str],
        summary: Dict,
        toxicity_markers: List[bool],
        whole_summary: str
    ) -> None:
        version = '1.0'
        title = transcription_title
        filename = os.path.basename(path_to_file)
        speakers = []
        for i in range(len(label_to_name)):
            cur_speaker = label_to_name[i] if label_to_name[i] is not None else f'Спикер {i+1}'
            speakers.append({
                'label': i,
                'name': cur_speaker,
                'confidence': 0.9
            })
        utterances = []
        for item in resulted_track:
            utterances.append({
                'started_at': item['start'],
                'finished_at': item['end'],
                'speaker': {
                    "label": item['speaker_label'],
                    "confidence": 0.85,
                    "mixed": 0
                },
                'text': item['text']
            })
        result = {
            'version': version,
            'title': title,
            'origin': {
                'file': {
                    'name': filename
                }
            },
            'speakers': speakers,
            'utterances': utterances,
            'summary': summary,
            'toxicity': toxicity_markers,
            'whole_summary': whole_summary
        }
        return result