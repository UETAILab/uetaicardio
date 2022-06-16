/*
 * Copyright 2020 UET-AILAB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ailab.aicardiotrainer.annotationscreen

import android.graphics.BitmapFactory
import android.util.Log
import androidx.lifecycle.viewModelScope
import com.ailab.aicardiotrainer.LCE
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.converJSONObjectOldVerion
import com.ailab.aicardiotrainer.interfaces.LoadingProgressListener
import com.ailab.aicardiotrainer.repositories.*
import com.imebra.DataSet
import kotlinx.android.synthetic.main.activity_annotation.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

class ReadDicomFileMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: ReadDicomFileMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: ReadDicomFileMVI()
                        .also { instance = it }
            }
        const val TAG = "ReadDicomFileMVI"
        fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
            getInstance().process(annotationActVM, annotationViewEvent)
        }

        fun renderViewState(annotationActivity: AnnotationActivity, viewModel: AnnotationActVM, viewState: AnnotationViewState) {
            getInstance().renderViewState(annotationActivity, viewModel, viewState)
        }

        fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
            getInstance().renderViewEffect(annotationActivity, viewEffect)
        }
    }

    private val dicomRepository = DicomRepository.getInstance()

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
        when(viewEffect) {
            is AnnotationViewEffect.LoadingProgress -> {
                annotationActivity.iv_draw_canvas.loadingText = "Loading to frame: ${viewEffect.progress}"
                annotationActivity.iv_draw_canvas.invalidate()
            }
        }
    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewModel: AnnotationActVM, viewState: AnnotationViewState) {

        if (AnnotationActivity.bitmapHeart == null) AnnotationActivity.bitmapHeart = BitmapFactory.decodeResource(annotationActivity.resources, R.drawable.heart)

        when(viewState.status) {
            AnnotationViewStatus.Fetched -> {
                Log.w(TAG, "AnnotationViewStatus.Fetched")

//                renderLoginButton(viewState.phone)
                val frameList = viewState.bitmaps.mapIndexed { index, bitmap ->
                    FrameItem(index=index, bitmap = bitmap)
                }
                annotationActivity.newsRvFrameAdapter.submitList(frameList)

                annotationActivity.newsRvFrameAdapter.setCurrentPosition(viewModel.getCurrentFrameIndex())

                annotationActivity.iv_draw_canvas.loadingText = ""

                if (viewState.bitmaps.size > 0) {
                    annotationActivity.iv_draw_canvas.setFitScale(viewState.bitmaps.get(0))
                }
//                Log.w(TAG,"Before getRenderAnnotationFrame")
                viewModel.getRenderAnnotationFrame()?.let {
                    Log.w(TAG, "${it.renderAnnotation}")

                    PlaybackMVI.renderViewEffect(annotationActivity, AnnotationViewEffect.RenderAnnotationFrame(it.renderAnnotation))

//                    Log.w(TAG,"After getRenderAnnotationFrame")

                }

//                annotationActivity.iv_draw_canvas.invalidate()


//                annotationActivity.rv_folder_list.scrollToPosition(currentPosition)
//
//                Log.w(TAG, "AnnotationViewStatus.Fetched ${currentPosition}")

//                SetToolUsingMVI.renderViewEffect(annotationActivity, AnnotationViewEffect.RenderDiagnosisTool(viewState.dicomDiagnosis))

            }

            AnnotationViewStatus.Fetching -> {

                Log.w(AnnotationActivity.TAG, "is async fetching")

                annotationActivity.iv_draw_canvas.setCustomImageBitmap(AnnotationActivity.bitmapHeart)
            }
        }
    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        when (annotationViewEvent) {

            is AnnotationViewEvent.FetchNewsFile -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(FetchNewsFileReducerAsync(annotationActVM, it, annotationViewEvent))
                }
            }

            is AnnotationViewEvent.NewProgess -> {
                annotationActVM.reduce(AnnotationStateEffectObject(null, AnnotationViewEffect.LoadingProgress(annotationViewEvent.progress)))
            }



            is AnnotationViewEvent.FileFetchedError -> {
                annotationActVM.reduce(AnnotationStateEffectObject(
                    annotationActVM.viewStates().value?.copy(status = AnnotationViewStatus.Fetched, bitmaps = emptyList()),
                    AnnotationViewEffect.ShowToast(message = annotationViewEvent.resultLaunch.message)
                ))
            }

            is AnnotationViewEvent.FileFetchedSuccess -> {
                Log.w(TAG, "GO BEFORE AnnotationViewEvent.FileFetchedSuccess")

//                annotationActVM.reduceStateEffectObject(annotationViewEvent.viewState, annotationViewEvent.viewEffect)
//                annotationViewEvent.resultLaunch.
                annotationActVM.reduce(annotationViewEvent.annotationStateEffectObject)

                val viewState = annotationViewEvent.annotationStateEffectObject.viewState
                viewState?.bitmaps.let {
                    it?.let {
                        if (it.size > 0) annotationActVM.setCurrentFrameIndex(0)
                    }
                }
//                if (annotationViewEvent.annotationStateEffectObject.viewState.bitmaps.size > 0) {
////                    Log.w(TAG, "annotationActVM size: ${annotationActVM.numFrame}")
//                    annotationActVM.setCurrentFrameIndex(0)
//                }

                Log.w(AnnotationActVM.TAG, "GO AFTER AnnotationViewEvent.FileFetchedSuccess")
            }

        }
    }



    inner class FetchNewsFileReducerAsync(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.FetchNewsFile)
        : AnnotationActReducer(viewModel, viewState, viewEvent), LoadingProgressListener {


        override fun reduce(): AnnotationStateEffectObject {

            Log.w(TAG, "FetchNewsFileReducerAsync ${viewEvent.file}")

            if (viewState.status == AnnotationViewStatus.Fetching) return AnnotationStateEffectObject()

            val file = viewEvent.file
            val result = AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.Fetching, file = file, bitmaps = emptyList(), dataset = DataSet(),
                    dicomAnnotation = DicomAnnotation(), dicomDiagnosis = DicomDiagnosis(),
                    machineAnnotation = DicomAnnotation(), machineDiagnosis = DicomDiagnosis(), tagsDicom = JSONObject()
                ),
                AnnotationViewEffect.ShowToast(message = "file ${file} fetching"))

            Log.w(TAG, "start file fetching")
            val listener : LoadingProgressListener = this

            //
            viewModel.viewModelScope.launch {
                Log.w(TAG, "in async")

                when (val readResult = dicomRepository.getDatasetAndBitmaps(file, listener)) {

                    is LCE.Result -> if (readResult.error) {
                        Log.w(TAG, "after call to getDatasetAndBitmaps fail")

                        viewModel.viewStates().value?.let {
                            ReadDicomFileMVI.process(viewModel, AnnotationViewEvent.FileFetchedError(readResult))
                        }

                    } else {
                        viewModel.viewStates().value?.let {
                            viewModel.reduce(GetAnnotationAndDiagnosisReducerAsync(viewModel, it, AnnotationViewEvent.AnnotationAndDiagnosis(readResult, file)))
                        }
                    }
                }
            }

            return result
        }

        override suspend fun onProgress(progress: Long) = withContext(Dispatchers.Main) {
            ReadDicomFileMVI.process(viewModel, AnnotationViewEvent.NewProgess(progress))
        }

    }

    inner class GetAnnotationAndDiagnosisReducerAsync(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.AnnotationAndDiagnosis) : AnnotationActReducer(viewModel, viewState, viewEvent) {

        private val annotationRepository = AnnotationRepository.getInstance()
        val bitmaps get() = viewEvent.result.data.bitmaps
        val dataset get() = viewEvent.result.data.dataset
        val tags get() = viewEvent.result.data.tags

        override fun reduce(): AnnotationStateEffectObject {
            val fileDicom = viewEvent.file
            Log.w(TAG, "start loadAnnotationAndDiagnosis ${fileDicom}")

            val fileJSON = fileDicom + ".json"
            val file = File(fileJSON)

            // MUST add new field if data strucutre in each frame change (DicomAnnotation)
            val viewStateIfError = viewState.copy(
                status = AnnotationViewStatus.Fetched,
                dataset = dataset,
                bitmaps = bitmaps,
                file = fileDicom,
                dicomAnnotation = DicomAnnotation.getNewAnnotation(bitmaps.size),
                dicomDiagnosis = DicomDiagnosis(),
                tagsDicom = tags,
                machineAnnotation = DicomAnnotation.getNewAnnotation(bitmaps.size),
                machineDiagnosis = DicomDiagnosis()
            )

            // HACK frame index
            if (viewStateIfError.bitmaps.size > 0) viewModel.forceCurrentFrameIndex(0)

            if (!file.exists()) {
                return AnnotationStateEffectObject(viewStateIfError, null)
            }

            viewModel.viewModelScope.launch {
                Log.w(TAG, "start launch read file ${fileJSON}")
                when (val readResult = annotationRepository.getAnnotationFromFile(fileJSON)) {

                    is LCE.Result -> {
                        if (readResult.error) {

                            ReadDicomFileMVI.process(viewModel,
                            AnnotationViewEvent.FileFetchedSuccess(
                                AnnotationStateEffectObject(viewStateIfError,
                                AnnotationViewEffect.ShowToast(message = readResult.message))))

                        } else {
//                        Log.w(TAG, "start launch read file ${readResult.error} ${readResult.message} ${readResult.data}$")

                            val effect = AnnotationViewEffect.ShowToast(message = readResult.message)

                            val manual_annotation = if(readResult.data.has(AnnotationActVM.MANUAL_ANNOTATION)) DicomAnnotation(readResult.data.getJSONArray(
                                AnnotationActVM.MANUAL_ANNOTATION
                            )) else viewStateIfError.dicomAnnotation

                            val manual_diagnosis = if(readResult.data.has(AnnotationActVM.MANUAL_DIAGNOSIS)) DicomDiagnosis(readResult.data.getJSONObject(
                                AnnotationActVM.MANUAL_DIAGNOSIS
                            ).toString()) else DicomDiagnosis()

                            val machine_annotation = if(readResult.data.has(AnnotationActVM.MACHINE_ANNOTATION)) DicomAnnotation(readResult.data.getJSONArray(
                                AnnotationActVM.MACHINE_ANNOTATION
                            )) else viewStateIfError.machineAnnotation

                            val machine_diagnosis = if(readResult.data.has(AnnotationActVM.MACHINE_DIAGNOSIS)) DicomDiagnosis(readResult.data.getJSONObject(
                                AnnotationActVM.MACHINE_DIAGNOSIS
                            ).toString()) else DicomDiagnosis()

                            if (manual_diagnosis.getInt(DicomDiagnosis.CHAMBER_IDX) == -1)
                                manual_diagnosis.put(DicomDiagnosis.CHAMBER_IDX, DicomDiagnosis.getChamberIdxFromName(manual_diagnosis.getString(DicomDiagnosis.CHAMBER)))


                            if (readResult.data.has("version")) { // VERSION
                                Log.w("annotationConvert", "${readResult.data}")
                                val annotationConvert = converJSONObjectOldVerion(readResult.data, nFrame = bitmaps.size, tags=tags)
                                val oldDiagnosis = annotationConvert.second

                                val state = viewStateIfError.copy( dicomAnnotation = annotationConvert.first, dicomDiagnosis = oldDiagnosis)


                                viewState.dicomAnnotation.updateLengthAreaVolumeAllFrame(tags)

                                ReadDicomFileMVI.process(viewModel, AnnotationViewEvent.FileFetchedSuccess(AnnotationStateEffectObject(state, effect) ))

                            } else {
                                val state = viewStateIfError.copy(dicomAnnotation = manual_annotation, dicomDiagnosis = manual_diagnosis, machineAnnotation = machine_annotation, machineDiagnosis = machine_diagnosis)
                                ReadDicomFileMVI.process(viewModel, AnnotationViewEvent.FileFetchedSuccess(AnnotationStateEffectObject(state, effect) ))

                            }

                        }

                    }
                }
            }

            return AnnotationStateEffectObject()
        }

    }
}