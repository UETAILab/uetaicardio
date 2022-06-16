package com.ailab.aicardio.annotationscreen

import android.app.Application
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import com.ailab.aicardio.R
import com.ailab.aicardio.getFileJSONFromResources
import com.ailab.aicardio.repository.*
import com.ailab.aicardio.repository.DicomAnnotation.Companion.AREA
import com.ailab.aicardio.repository.DicomAnnotation.Companion.EF_BOUNDARY
import com.ailab.aicardio.repository.DicomAnnotation.Companion.EF_POINT
import com.ailab.aicardio.repository.DicomAnnotation.Companion.GLS_BOUNDARY
import com.ailab.aicardio.repository.DicomAnnotation.Companion.GLS_POINT
import com.ailab.aicardio.repository.DicomAnnotation.Companion.LENGTH
import com.ailab.aicardio.repository.DicomAnnotation.Companion.VOLUME
import com.ailab.aicardio.repository.EFObject.Companion.convertToEFObject
import com.imebra.DataSet
import com.rohitss.aacmvi.AacMviViewModel
import org.json.JSONArray
import org.json.JSONObject
import java.io.File


@RequiresApi(Build.VERSION_CODES.N)
class AnnotationActVM(val applicationAnnotate: Application) :
    AacMviViewModel<AnnotationViewState, AnnotationViewEffect, AnnotationViewEvent>(applicationAnnotate) {

    companion object {

        const val TAG = "AnnotationActVM"
        const val DICOM_TAG = "dicom_tags"
        const val PHONE = "deviceID"
        const val FILE = "path_file"
        const val MANUAL_DIAGNOSIS = "dicomDiagnosis"
        const val MANUAL_ANNOTATION = "dicomAnnotation"
        const val MACHINE_ANNOTATION = "machineAnnotation"
        const val MACHINE_DIAGNOSIS= "machineDiagnosis"
        const val KEY_VERSION = "VERSION"
        const val VERSION_NUMBER = "4.0"
    }


    init {
        viewState = AnnotationViewState(
            status = AnnotationViewStatus.NotFetched,
            dataset = DataSet(),
            folderList = emptyList(),
            folder = "",
            phone = User.getPhone(context = applicationAnnotate.applicationContext)?:User.DEFAULT_PHONE,
            bitmaps = emptyList(),
            file = "",
            dicomAnnotation = DicomAnnotation(),
            dicomDiagnosis = DicomDiagnosis(),
            tagsDicom = JSONObject(),
            machineAnnotation = DicomAnnotation(),
            machineDiagnosis = DicomDiagnosis(),
            boundaryHeart = getFileJSONFromResources(applicationAnnotate.resources, R.raw.heart_convex)
        )
    }

    val boundaryHeart get() = viewState.boundaryHeart
    val numFrame get() =  viewState.bitmaps.size

    private var currentFrameIndex: Int = 0
    private var isPlaying: Boolean = true

    private var currentToolId: Int? = null
    private var isGls: Boolean = false // check value at Perimeter checkbox, if it true then draw for gls else ef

    private var enableManualDraw: Boolean = true
    private var enableAutoDraw: Boolean = false

    private var toolClickedType: String? = null
    private var modifyPointIndex = -1

    private var modifyBoundaryIndex = Pair(-1, -1) // (index of boundary, index of point)
    private val isValidFrameState get() = (isPlaying == false && numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame)

    private val isESV get() = viewState.dicomAnnotation.getIsESVWithFrameIndex(currentFrameIndex)
    private val isEDV get() = viewState.dicomAnnotation.getIsEDVWithFrameIndex(currentFrameIndex)

    private val lengthCalculator get() = viewState.dicomAnnotation.getMeasureByKey(currentFrameIndex, LENGTH)
    private val areaCalculator get() = viewState.dicomAnnotation.getMeasureByKey(currentFrameIndex, AREA)
    private val volumeCalculator get() = viewState.dicomAnnotation.getMeasureByKey(currentFrameIndex, VOLUME)
    val nPointEF get() = viewState.dicomAnnotation.getPointArray(currentFrameIndex, EF_POINT).length()
    val nPointGLS get() = viewState.dicomAnnotation.getPointArray(currentFrameIndex, GLS_POINT).length()
    val textEsvEDV get() = viewState.dicomAnnotation.getEsvEDVTextDraw()


    private val keyModEfGls get() = if (isGls == true) DicomAnnotation.MOD_GLS else DicomAnnotation.MOD_EF

    val keyPoint get() = "${keyModEfGls}${DicomAnnotation.MOD_POINT}"
    val keyBoundary get() = "${keyModEfGls}${DicomAnnotation.MOD_BOUNDARY}"

    fun getIsValidFrameState(): Boolean {
        return isValidFrameState
    }

    fun getCurrentTool(): Pair<Int?, String?> {
        return Pair(currentToolId, toolClickedType)
    }

    fun reduce(reducer: AnnotationActReducer) {
        val result = reducer.reduce()
        result.annotationViewState?.let { viewState = it }
        result.annotationViewEffect?.let { viewEffect = it }
    }

    fun reduceStateEffectObject(newViewState: AnnotationViewState?, newViewEffect: AnnotationViewEffect?) {
        newViewState?.let { viewState = it }
        newViewEffect?.let { viewEffect = it}
    }

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


    override fun process(viewEvent: AnnotationViewEvent) {
        super.process(viewEvent)
        when (viewEvent) {

            is AnnotationViewEvent.NewsItemFileClicked -> newsItemFileClicked(viewEvent.folderItem)
            is AnnotationViewEvent.NewsItemFileLongClicked -> newsItemFileClicked(viewEvent.folderItem)

            is AnnotationViewEvent.NewsFrameClicked -> newsFrameClicked(viewEvent.frameItem)
//            is AnnotationViewEvent.NewsFrameLongClicked -> newsFrameLongClicked(viewEvent.frameItem)

        }
    }

    private fun newsFrameClicked(frameItem: FrameItem) {
        SaveDataMVI.process(this, AnnotationViewEvent.OnSaveDataToDisk())
        currentFrameIndex = frameItem.index
        getRenderAnnotationFrame()?.let { viewEffect = it }
    }

    private fun newsItemFileClicked(folderItem: FolderItem) {
        val file = File(folderItem.path)

        if (file.isFile) {
            // NOTE: safe viewModel using currentFrame of old file
            if (file.absolutePath != viewState.file) {

                setCurrentFrameIndex(-1)
//            reduce(FetchNewsFileReducerAsync(this, viewState, AnnotationViewEvent.FetchNewsFile(file.absolutePath)))
                FetchNewsFileMVI.process(this, AnnotationViewEvent.FetchNewsFile(file.absolutePath))
            }
        } else {

            if (file.absolutePath != viewState.folder)
                viewState = viewState.copy(status = AnnotationViewStatus.OpenAnnotationActivity(folder = file.absolutePath))
        }
    }

    fun getArrayFolderItem(): List<FolderItem> {
        return viewState.folderList
    }


    fun getRenderAnnotationFrame(): AnnotationViewEffect.RenderAnnotationFrame? {
//        if (numFrame > 0) return AnnotationViewEffect.RenderAnnotationFrame(RenderAnnotation())
//        Log.w("getRenderAnnotationFrame", "c: $currentFrameIndex #: $numFrame")

        return if (currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
            val o = viewState.dicomAnnotation.getFrameAnnotationObj(currentFrameIndex)
            val ro = RenderAnnotation(nPointGLS = nPointGLS, nPointsEF = nPointEF, ef = convertToEFObject(o), esvEdvText = textEsvEDV,
                length = lengthCalculator, area =  areaCalculator, volume = volumeCalculator,
                isPlaying = isPlaying, numFrame = numFrame, indexFrame = currentFrameIndex,
                infoText = getInfoText(), bitmap = getCurrentFrameBitmap(), isESV=isESV, isEDV=isEDV)
            return AnnotationViewEffect.RenderAnnotationFrame(ro)
        } else null
    }

    private fun getCurrentFrameBitmap(): Bitmap {
        return viewState.bitmaps.get(currentFrameIndex)
    }

    fun getInfoText(): String {
        return "${viewState.dicomAnnotation.getNFrameAnnotated()} / ${currentFrameIndex + 1} / ${numFrame} / ${File(viewState.file).name}"

    }

    fun getFileName(): String {
        return viewState.file
    }

    fun getDiagnosis(): DicomDiagnosis {
        return viewState.dicomDiagnosis
    }

    fun getIsPlaying(): Boolean {
        return isPlaying
    }

    fun getListFrameList(): List<FrameItem> {
        return viewState.bitmaps.mapIndexed { index, bitmap ->
            FrameItem(index,  getResizedBitmap(bitmap, 50, 35) )
        }
    }

    fun getResizedBitmap(bm: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        val width = bm.width
        val height = bm.height
        val scaleWidth = newWidth.toFloat() / width
        val scaleHeight = newHeight.toFloat() / height
        // CREATE A MATRIX FOR THE MANIPULATION
        val matrix = Matrix()
        // RESIZE THE BIT MAP
        matrix.postScale(scaleWidth, scaleHeight)

        // "RECREATE" THE NEW BITMAP
        return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false)
    }

    fun hasNoLabel(): Boolean {
        return (viewState.dicomDiagnosis.chamberIdx == -1)
    }

    fun setIsGls(value: Boolean) {
        this.isGls = value
    }


    fun getIsGls(): Boolean? {
        return isGls
    }

    fun setIsPlaying(isPlayingValue: Boolean) {
        isPlaying = isPlayingValue
    }

    fun setCurrentToolId(toolId: Int?) {
        this.currentToolId = toolId
    }

    fun setToolClickedType(typeClicked: String) {
        this.toolClickedType = typeClicked
    }

    fun getCurrentFrameIndex(): Int {
        return currentFrameIndex
    }

    fun hasEFBoundaryAndPoint(frameIdx: Int): Boolean {
//        Log.w(TAG, "hasEFBoundaryAndPoint -- ${frameIdx}")
        return viewState.dicomAnnotation.hasBoundaryAndPoint(frameIdx, key_point = EF_POINT, key_boundary = EF_BOUNDARY)
    }

    fun hasGLSBoundaryAndPoint(frameIdx: Int): Boolean {
        return viewState.dicomAnnotation.hasBoundaryAndPoint(frameIdx, key_point = GLS_POINT, key_boundary = GLS_BOUNDARY)
    }

    fun setCurrentFrameIndex(frameIdx: Int) {
//        Log.w("setCurrentFrameIndex", "$frameIdx")
        if (numFrame > 0)
            currentFrameIndex = frameIdx
        else currentFrameIndex = -1
    }

    fun forceCurrentFrameIndex(frameIdx: Int) {
        currentFrameIndex = frameIdx
    }

    fun setAutoDraw(autoDrawMachine: Boolean) {
        enableAutoDraw = autoDrawMachine
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

}