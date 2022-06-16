package com.ailab.aicardiotrainer.nav_interpretation.ui.annotation

import aacmvi.AacMviFragment
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.RecyclerView
import com.ailab.aicardiotrainer.R
import kotlinx.android.synthetic.main.fragment_annotation.*

class AnnotationFragment : AacMviFragment<InterpretationViewState, InterpretationViewEffect, InterpretationViewEvent, InterpretationActVM>(),
    OnDrawListener,
    OnNormalizeTouchListener,
    OnSopInstanceUIDItemClicked {



    companion object {

        // For Singleton instantiation
        @Volatile
        private var instance: AnnotationFragment? = null

        const val TAG = "AnnotationFragment"
        const val MY_PERMISSIONS_REQUEST_CODE = 1

        var bitmapHeart : Bitmap? = null
        var bitmapPlay : Bitmap? = null
        var bitmapPause : Bitmap? = null




    }
    override val viewModel: InterpretationActVM by viewModels()

//    override fun onAttach(context: Context) {
//        super.onAttach(context)
//        context?.let {
//
//        }
//    }
//

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
//        val test_tool_06 = view.findViewById<Button>(R.id.test_tool_06)
//    test_tool_01.setText("TEST-TUANNM")
//
        test_tool_06.setOnClickListener {
            val studyInstanceUID = "1.2.40.0.13.0.11.2672.5.2013102492.1340595.20130717095716"
            StudyRepresentationMVI.process(viewModel, InterpretationViewEvent = InterpretationViewEvent.LoadingRepresentationStudyInstanceUID(studyInstanceUID = studyInstanceUID))
        }

//        val inter_gv_dicom_preview = view.findViewById<GridView>(R.id.inter_gv_dicom_preview)
//        val inter_iv_draw_canvas = view.findViewById<InterpretationCanvasView>(R.id.inter_iv_draw_canvas)
//        val inter_rv_frames = view.findViewById<RecyclerView>(R.id.inter_rv_frames)
//        val bt_prev_frame = view.findViewById<ImageButton>(R.id.bt_prev_frame)
//        val bt_next_frame = view.findViewById<ImageButton>(R.id.bt_next_frame)
//        val bt_play_pause = view.findViewById<ImageButton>(R.id.bt_play_pause)
//        val inter_seek_bar = view.findViewById<SeekBar>(R.id.inter_seek_bar)


        studyRepresentationGVAdapter?.let {
            inter_gv_dicom_preview.apply {
                adapter = it
            }
        }

        inter_iv_draw_canvas.setOnDrawListener(this)
        inter_iv_draw_canvas.setOnNormalizeTouchListener(this)


        inter_rv_frames.adapter = interpretationFrameRVAdapter


        bt_prev_frame.setOnLongClickListener {
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowFirstFrame)
            true
        }

        bt_next_frame.setOnLongClickListener{
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowLastFrame)
            true
        }

        bt_next_frame.setOnClickListener {
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowNextFrame)
        }

        bt_prev_frame.setOnClickListener {
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowPreviousFrame)
        }

        bt_play_pause.setOnClickListener {
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.PlayPauseVideo)
        }

        inter_seek_bar?.let {
            it.setOnSeekBarChangeListener(
                object : SeekBar.OnSeekBarChangeListener {
                    override fun onProgressChanged(seek: SeekBar, progress: Int, fromUser: Boolean) {
                        //                    InterpretationEnhanceBitmapMVI.process(viewModel, InterpretationViewEvent.EnhanceContrastBitmap(threshold=seek.progress))

                            viewModel.process(InterpretationViewEvent.OnChangeGammaCorrection(threshold = seek.progress))

                            renderTextViewGammaCorrection()


                    }

                    override fun onStartTrackingTouch(seek: SeekBar) {}

                    override fun onStopTrackingTouch(seek: SeekBar) {
                        Log.w(TAG, "Progress: ${seek.progress}")
                        //                    InterpretationEnhanceBitmapMVI.process(viewModel, InterpretationViewEvent.EnhanceContrastBitmap(threshold=seek.progress))
                    }
                }
            )
        }


    }


    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {

        super.onCreateView(inflater, container, savedInstanceState)

        val view = inflater.inflate(R.layout.fragment_annotation, container, false)
        return view

    }



    override fun onStart() {
        super.onStart()
        pushVideoToCanvas(handle)

        // render gridview of list file dicom
        viewModel.viewStates().value?.let {
            studyRepresentationGVAdapter?.submitList(it.getListSopInstanceUIDItem())
        }

        // render frame rv adapter of SopinstanceUID (dicom file)
        viewModel.viewStates().value?.let {
            interpretationFrameRVAdapter.submitList(it.getSubmitListFrameItem())
        }
        // render canvas
        viewModel.getRenderMP4FrameObject()?.let {
            InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderMP4Frame(it))
        }

        // render component view in activity
        InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderComponentActivity(idComponent = R.id.bt_play_pause, isPlaying = viewModel.getIsPlaying()))

    }

    val handle =  Handler()



    val interpretationFrameRVAdapter by lazy {
        InterpretationFrameRVAdapter(listener = {

            viewModel.process(InterpretationViewEvent.SopInstanceUIDFrameClicked(it.tag as FrameItem))


        }, longListener = {

            viewModel.process(InterpretationViewEvent.SopInstanceUIDFrameLongClicked(it.tag as FrameItem))

            true
        }, isVertical = true, interpretationActVM = viewModel)
    }




    private fun pushVideoToCanvas(handler: Handler) {
        handler.postDelayed({
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.NextFrame)
            pushVideoToCanvas(handler)
        }, 30L)
    }

    val studyRepresentationGVAdapter by lazy {
        this.context?.let {
            StudyRepresentationGVAdapter(it, this)
        }
    }



    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        inter_iv_draw_canvas.saveToBundle(outState)
    }




    fun renderTextViewGammaCorrection() {

        InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderTextViewGammaCorrection(viewModel.getGammaValue()))
    }







    override fun renderViewState(viewState: InterpretationViewState) {
        Log.w(TAG, "renderViewState: ${viewState}")

        StudyRepresentationMVI.renderViewState(this, viewState)
        SopInstanceUIDMVI.renderViewState(this, viewState)

        InterpretationEnhanceBitmapMVI.renderViewState(this, viewState)

        if (bitmapHeart == null) {
            bitmapHeart = BitmapFactory.decodeResource(this.resources, R.drawable.heart)
        }

        when(viewState.status) {
//            InterpretationViewStatus.Start -> {
//
////                Log.w(TAG, "Go InterpretationViewStatus.Start ${bitmapHeart?.width} ${bitmapHeart?.height}")
////                inter_iv_draw_canvas.setCustomImageBitmap(bitmapHeart)
//                val studyInstanceUID = "1.2.40.0.13.0.11.2672.5.2013102492.1340595.20130717095716"
//                StudyRepresentationMVI.process(viewModel, InterpretationViewEvent = InterpretationViewEvent.LoadingRepresentationStudyInstanceUID(studyInstanceUID = studyInstanceUID))
//
//            }

        }
    }

    override fun renderViewEffect(viewEffect: InterpretationViewEffect) {
        Log.w(TAG, "renderViewEffect: ${viewEffect}")
        InterpretationPlaybackMVI.renderViewEffect(this, viewEffect)

        when (viewEffect) {

            is InterpretationViewEffect.ShowToast -> this.context?.let{
                it.toast(message = viewEffect.message)

            }
        }

    }

    override fun onTouchEvent(view: InterpretationCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
        Log.w(TAG, "onTouchEvent ${event?.action} ${ix} ${iy}")
    }

    override fun draw(view: InterpretationCanvasView, canvas: Canvas?) {
//        TODO("Not yet implemented")
    }

    override fun onSopInstanceUIDItemClicked(item: SopInstanceUIDItem) {
        Log.w(TAG, "onSopInstanceUIDItemClicked ${item}")

    }

    override fun onSopInstanceUIDItemLongClicked(item: SopInstanceUIDItem) {
        Log.w(TAG, "onSopInstanceUIDItemLongClicked ${item}")
        SopInstanceUIDMVI.process(viewModel, InterpretationViewEvent.LoadingMP4SopInstanceUID(item))

    }
}