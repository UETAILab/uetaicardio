/*
 * Copyright 2021 UET-AILAB
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

package com.uetailab.aipacs.home_pacs.fragment_intepretation
import aacmvi.AacMviViewModel
import android.app.Application
import android.graphics.Bitmap
import android.util.Log
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.fragment_home.HomeViewVM
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject

class InterpretationViewVM(val applicationInterpretation: Application) : AacMviViewModel<InterpretationViewState, InterpretationViewEffect, InterpretationViewEvent>(applicationInterpretation) {
    companion object {
        const val TAG = "InterpretationViewVM"
    }
    var cntClicked = 0
    init {
        viewState = InterpretationViewState(
            status = InterpretationViewStatus.Start,
            message = "Start-InterpretationView-Activity",
            studies = emptyList(),
            studyMetaData = JSONObject(),
            studyPreview = emptyList(),
            bitmaps = emptyList(),
            studyInterpretation = JSONObject(),
            dicomInterpretation = DicomInterpretation(),
            machineInterpretation = DicomInterpretation(getFileJSONFromResourcesJSONObject(applicationInterpretation.applicationContext.resources, R.raw.result)),
            dicomMetaData = JSONObject(),
            boundaryHeart = getFileJSONFromResourcesJSONArray(applicationInterpretation.applicationContext.resources, R.raw.heart_convex),
            efObject = EFObject()
        )
    }
    val numFrame get() =  viewState.bitmaps.size
    val efObject get() = viewState.efObject

    private var currentFrameIndex: Int = 0
    private var isPlaying: Boolean = true

    private val isESV get() = viewState.dicomInterpretation.getIsESVWithFrameIndex(currentFrameIndex)
    private val isEDV get() = viewState.dicomInterpretation.getIsEDVWithFrameIndex(currentFrameIndex)

    val dicomDiagnosis get() = viewState.dicomDiagnosis
    val boundaryHeart get() = viewState.boundaryHeart

    val studyInterpretation get() = viewState.studyInterpretation

    private var toolNameUsing : String = ""
    private var toolClickedType : Boolean = false // false: short clicked/, true: long clicked
    private var modifyPointIndex = -1

    private var modifyBoundaryIndex = Pair(-1, -1) // (index of boundary, index of point)

    private var isGls: Boolean = false // check value at Perimeter checkbox, if it true then draw for gls else ef

    private var enableManualDraw: Boolean = true
    private var enableAutoDraw: Boolean = false


    fun getModifyPointIndex(): Int {
        return modifyPointIndex
    }
    fun setModifyPointIndex(value: Int) {
        modifyPointIndex = value
    }

    fun setModifyBoundaryIndex(value: Pair<Int, Int>) {
        modifyBoundaryIndex = value
    }

    fun getModifyBoundaryIndex(): Pair<Int, Int> {
        return modifyBoundaryIndex
    }


    fun isStartEmptyFragment(): Boolean {
        return viewState.status is InterpretationViewStatus.Start
    }

    fun setCurrentFrameIndexWithClick(buttonID: Int, isLongClicked: Boolean, isPlayingNextFrame: Boolean) {
//        Log.w(TAG, "setCurrentFrameIndexWithClick: ${currentFrameIndex} isPlayingNextFrame: ${isPlayingNextFrame} ${isLongClicked}")
        if (numFrame > 0) {
            when (buttonID) {
                R.id.bt_prev_frame -> {
                    isPlaying = false
                    if (isLongClicked) currentFrameIndex = 0
                    else currentFrameIndex = if (currentFrameIndex == 0) numFrame - 1 else currentFrameIndex - 1

                }
                R.id.bt_next_frame -> {
//                    Log.w(TAG, "setCurrentFrameIndexWithClick --- bt_next_frame: ${currentFrameIndex} isPlayingNextFrame: ${isPlayingNextFrame} ${isLongClicked}")
                    if (isPlayingNextFrame) {
                        if (isPlaying == true) {
                            if (isLongClicked) currentFrameIndex = numFrame - 1
                            else currentFrameIndex = if (currentFrameIndex >= numFrame - 1) 0 else currentFrameIndex + 1
                        }
                    } else {
                        isPlaying = false
                        if (isLongClicked) currentFrameIndex = numFrame - 1
                        else currentFrameIndex = if (currentFrameIndex >= numFrame - 1) 0 else currentFrameIndex + 1
                    }

                }
                R.id.bt_play_pause -> {
                    isPlaying = !isPlaying
                }
            }
            if (numFrame == 1) isPlaying = false

        }
//        Log.w(TAG, "setCurrentFrameIndexWithClick: ${currentFrameIndex} ${isPlaying}")
    }


    fun reduce(reducer: InterpretationViewReducer) {
        val result = reducer.reduce()
        reduce(result)
    }


    fun reduce(result: InterpretationViewObject) {
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }

    override fun process(viewEvent: InterpretationViewEvent) {
        super.process(viewEvent)
        when (viewEvent) {
            is InterpretationViewEvent.FrameCanvasClicked -> {
               if (numFrame > 0) {
                   currentFrameIndex = viewEvent.frameCanvasItem.index
                   isPlaying = false
                   reduce(InterpretationViewObject(viewState = null, viewEffect = getRenderFragmentViewEffect()))
               }
            }
        }

    }

    val studyID get() = viewState.studyID
    val studyInstanceUID get() = viewState.studyInstanceUID
    val studyPreview get() = viewState.studyPreview
    val studyMetadata get() = viewState.studyMetaData
    val listFileDicom: JSONObject get() = if (studyMetadata.has("ListFileDicom")) studyMetadata.getJSONObject("ListFileDicom") else JSONObject()
    val relativePath get() = viewState.relativePath
    val dicomMetadata get() = viewState.dicomMetaData
    fun getDicomViewAndNumberOfFrame(fileName: String): Pair<String, String> {
//        Log.w(TAG, "fileName: ${fileName} getDicomViewAndNumberOfFrame: ${newFileName}")
        listFileDicom.keys().forEach {
//            Log.w(TAG, "Key: ${it}")
            if (it.contains(fileName)) {
                try {
                    val obj = listFileDicom.getJSONObject(it)
                    return Pair(obj.getJSONObject("DicomView").getJSONObject("DataView").getString("View"), obj.getInt("NumberOfFrames").toString() + " --- " + obj.getInt("InstanceNumber").toString()  )
                } catch (e: Exception) {
                    Log.w(HomeViewVM.TAG, "getDicomViewAndNumberOfFrame: ${e}")
                    return Pair("No-DicomView", "1")
                }
            }
        }
        return Pair("No-DicomView", "1")
    }

    fun onPassDataFromHomeView(homeViewVM: HomeViewVM) {
        val value = homeViewVM.viewStates().value
        if (value != null) {

            isPlaying = value.bitmaps.size != 1
            Log.w(TAG, "onPassDataFromHomeView ${value.bitmaps.size }")

            var dicomInterpretation = DicomInterpretation(value.bitmaps.size)
            val machineInterpretation = DicomInterpretation(value.bitmaps.size)

            var dicomMetadata = JSONObject()

            val studyInterpretation = value.studyInterpretation // JSONObject()
            Log.w(TAG, "studyInterpretation ${studyInterpretation}")
            Log.w(TAG, "onPassDataFromHomeView ${value.relativePath} ${studyInterpretation.has(value.relativePath)}")

            value.relativePath?.let {
                if (!studyInterpretation.has(value.relativePath)) {
                    studyInterpretation.put(value.relativePath, dicomInterpretation)
                }
                else {
                    // GET value annotation from file
                    Log.w(TAG, "Get value annotation from relative path: ${value.relativePath}")
                    dicomInterpretation = DicomInterpretation(studyInterpretation.getJSONObject(value.relativePath))

                }
                dicomMetadata = InterpretationViewState.getDicomMetaDataFromJSONObject(value.studyMetaData, it)

                Log.w(TAG, "PassDicomMetaData ${dicomMetadata}")

            }


            reduce(InterpretationViewObject(
                viewState = viewState.copy(
                    studies = value.studies,
                    studyID = value.studyID,
                    studyInstanceUID = value.studyInstanceUID,
                    studyPreview = value.studyPreview,
                    studyMetaData = value.studyMetaData,
                    relativePath = value.relativePath,
                    bitmaps = value.bitmaps,
                    dicomMetaData = dicomMetadata,
                    status = InterpretationViewStatus.PassedDataFromHomeView,
                    dicomInterpretation = dicomInterpretation,
                    machineInterpretation = machineInterpretation,
                    studyInterpretation = studyInterpretation
                ),
                viewEffect = InterpretationViewEffect.ShowToast("Done passing DATA from HomeView")
            ))
        }

    }

    fun onPassDataFromInterpretationView(interpretationViewVM: InterpretationViewVM) {
        val value = interpretationViewVM.viewStates().value
        if (value != null) {

            isPlaying = value.bitmaps.size != 1

            var dicomInterpretation = value.dicomInterpretation
            val studyInterpretation = value.studyInterpretation
            value.relativePath?.let {
                if (studyInterpretation.has(value.relativePath)) {
                    dicomInterpretation = DicomInterpretation(studyInterpretation.getJSONObject(value.relativePath))
                }
            }

            reduce(InterpretationViewObject(
                viewState = viewState.copy(
                    studies = value.studies,
                    studyID = value.studyID,
                    studyInstanceUID = value.studyInstanceUID,
                    studyPreview = value.studyPreview,
                    studyMetaData = value.studyMetaData,
                    relativePath = value.relativePath,
                    bitmaps = value.bitmaps,
                    status = InterpretationViewStatus.PassedDataFromInterpretationView,
                    dicomInterpretation = dicomInterpretation,
                    machineInterpretation = value.machineInterpretation,
                    studyInterpretation = studyInterpretation,
                    dicomMetaData = value.dicomMetaData
                ),
                viewEffect = InterpretationViewEffect.ShowToast("Done passing DATA from InterpretationView")
            )
            )
        }
    }

    fun getTextDrawCanvas(): String {
        return "${currentFrameIndex + 1}/ ${numFrame}/ 0 ${getCurrentStudyName()}"
    }
    fun getCurrentStudyName(): String? {
        return viewState.relativePath?.substringAfterLast("____")
    }

    fun getCurrentFrameBitmap(): Bitmap? {
        return if (viewState.bitmaps.size > currentFrameIndex) viewState.bitmaps.get(currentFrameIndex) else null
    }

    fun getRenderFragmentViewEffect(): InterpretationViewEffect.RenderFragmentView? {
        if (currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
            val rfv = FragmentViewObject(numFrame = viewState.bitmaps.size, bitmap = getCurrentFrameBitmap(),
                frameID = currentFrameIndex, frameText = getTextDrawCanvas(), isESV=isESV, isEDV=isEDV, isPlaying = isPlaying)
            return InterpretationViewEffect.RenderFragmentView(rfv)
        }
        return null
    }


    fun getRelativePath(itemName: String): String? {
        val shortNameArr = itemName.split("____")
        val fileName = shortNameArr[shortNameArr.size - 1]
        listFileDicom.keys().forEach {
            if (it.contains(fileName)) {
                try {
                    val obj = listFileDicom.getJSONObject(it)
                    return obj.getString("RelativePath")
                } catch (e: Exception) {
                    Log.w(HomeViewVM.TAG, "getDicomViewAndNumberOfFrame: ${e}")
                    return null
                }
            }
        }
        return null

    }

    fun getIsPlaying(): Boolean {
        return isPlaying
    }
    fun setIsPlaying(valueIsPlaing: Boolean) {
        isPlaying = valueIsPlaing
    }

    fun getCurrentFrameIndex(): Int {
        return currentFrameIndex
    }

    var interpretationViewUsingTool: InterpretationViewTool = InterpretationViewTool.OnClickNoAction(imageView = null)

    fun setToolUsing(interpretationViewTool: InterpretationViewTool) {
        interpretationViewUsingTool = interpretationViewTool
    }

    fun getIsValidFrameState(): Boolean {
        return (numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame && isPlaying == false)
    }

    fun getCurrentInterpretationToolClick(): InterpretationViewTool {
        return interpretationViewUsingTool
    }

//    private val keyModEfGls get() = if (isGls == true) "gls" else "ef"
//
//    val keyPoint get() = "${keyModEfGls}${DicomAnnotation.MOD_POINT}"
//    val keyBoundary get() = "${keyModEfGls}${DicomAnnotation.MOD_BOUNDARY}"
//
    val keyDrawPoint: String
        get() = if (isGls) FrameAnnotation.GLS_POINT else FrameAnnotation.EF_POINT

    val keyDrawBoundary: String
        get() = if (isGls) FrameAnnotation.GLS_BOUNDARY else FrameAnnotation.EF_BOUNDARY

    val keyMeasureLength: String
        get() = FrameAnnotation.MEASURE_LENGTH

    val keyMeasureArea: String
        get() = FrameAnnotation.MEASURE_AREA

    fun setAutoDraw(autoDrawMachine: Boolean) {
        enableAutoDraw = autoDrawMachine
        Log.w(TAG, "machineInterpreation: ${viewState.machineInterpretation}")
    }

    fun setManualDraw(manualDraw: Boolean) {
        enableManualDraw = manualDraw
    }

    fun getEnableManualDraw(): Boolean {
        return enableManualDraw
    }

    fun getEnableAutoDraw(): Boolean {
        return enableAutoDraw
    }

    fun getIsGls(): Boolean {
        return isGls
    }

    fun setIsGls(value: Boolean) {
        isGls = value
    }

    fun getBitmapWithFrameID(frameID: Int): Bitmap? {
        return if (frameID >= 0 && frameID < numFrame) viewState.bitmaps.get(frameID) else null
    }
    // TODO
    fun getMetaDataFrame(frameID: Int): JSONObject {
        return viewState.dicomInterpretation.getDataFrame(frameID)
    }
    fun getBitmapESV(): Bitmap {
        return viewState.bitmaps.get(0)
    }
    fun getBitmapEDV(): Bitmap {
        return viewState.bitmaps.get(0)
    }

    fun getESV_EDV_ID(): Pair<Int, Int> {
        return Pair(0, 0)
    }

    suspend fun getEFValueFromAllFrame(): EFObject = withContext(Dispatchers.IO) {
        val dicomMetadata = viewState.dicomMetaData
        viewState.dicomInterpretation.updateLengthAreaVolumeAllFrame(dicomMetadata)
        return@withContext viewState.dicomInterpretation.getEFByVolumeAllFrame()
    }

    fun getGlsValue(): Float {
        return viewState.glsValue
    }

    fun hasEFBoundaryAndPoint(frameIdx: Int): Boolean {
//        Log.w(TAG, "hasEFBoundaryAndPoint -- ${frameIdx}")
        return viewState.dicomInterpretation.hasBoundaryAndPoint(frameIdx, key_point = FrameAnnotation.EF_POINT, key_boundary = FrameAnnotation.EF_BOUNDARY)
    }

    fun hasGLSBoundaryAndPoint(frameIdx: Int): Boolean {
        return viewState.dicomInterpretation.hasBoundaryAndPoint(frameIdx, key_point = FrameAnnotation.GLS_POINT, key_boundary = FrameAnnotation.GLS_BOUNDARY)
    }

    fun getHasAnnotation(relativePath: String): Boolean{
        return studyInterpretation.has(relativePath)
    }
}