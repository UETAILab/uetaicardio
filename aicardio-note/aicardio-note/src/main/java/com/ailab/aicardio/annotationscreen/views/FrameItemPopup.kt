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

package com.ailab.aicardio.annotationscreen.views

import android.annotation.SuppressLint
import android.app.ActionBar
import android.content.Context
import android.graphics.Rect
import android.graphics.drawable.BitmapDrawable
import android.os.Handler
import android.os.Message
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.PopupWindow
import com.ailab.aicardio.R
import kotlinx.android.synthetic.main.popup_frame_item.view.*


@Suppress("DEPRECATION")
class FrameItemPopup(context: Context) {
    companion object{
        private const val MSG_DISMISS_TOOLTIP = 100
    }

    private var ctx: Context = context
    private var tipWindow: PopupWindow? = null
    private var contentView: View
    private var inflater: LayoutInflater

    @SuppressLint("SetTextI18n")
    fun showPopup(anchor: View, id: Int = 0, numOfEF: Int=0, numOfGLS: Int=0){

        tipWindow?.height = ViewGroup.LayoutParams.WRAP_CONTENT
        tipWindow?.width = ViewGroup.LayoutParams.WRAP_CONTENT


//        tipWindow?.isOutsideTouchable  = true
//        tipWindow?.isTouchable = true
//        tipWindow?.isFocusable = true
        tipWindow?.setBackgroundDrawable(BitmapDrawable())

        contentView.tv_id.text = "Index: $id"
        contentView.tv_ef.text = "PointEF: $numOfEF"
        contentView.tv_gls.text = "PointGLS: $numOfGLS"

        tipWindow?.contentView = contentView

        val screenPos = IntArray(2)
        // Get location of anchor view on screen
        // Get location of anchor view on screen
        anchor.getLocationOnScreen(screenPos)

        // Get rect for anchor view
        // Get rect for anchor view
        val anchorRect = Rect(
            screenPos[0], screenPos[1], screenPos[0]
                    + anchor.width, screenPos[1] + anchor.height
        )

        // Call view measure to calculate how big your view should be.
        // Call view measure to calculate how big your view should be.
        contentView.measure(
            ActionBar.LayoutParams.WRAP_CONTENT,
            ActionBar.LayoutParams.WRAP_CONTENT
        )


        // In this case , i dont need much calculation for x and y position of
        // tooltip
        // For cases if anchor is near screen border, you need to take care of
        // direction as well
        // to show left, right, above or below of anchor view
                // In this case , i dont need much calculation for x and y position of
        // tooltip
        // For cases if anchor is near screen border, you need to take care of
        // direction as well
        // to show left, right, above or below of anchor view
        val positionX: Int = anchorRect.centerX() - 60
        val positionY: Int = anchorRect.top - anchorRect.height() + 20

        tipWindow?.animationStyle = R.style.popup_window_animation
        tipWindow?.showAtLocation(
            anchor, Gravity.NO_GRAVITY, positionX,
            positionY
        )

        // send message to handler to dismiss tipWindow after X milliseconds
        // send message to handler to dismiss tipWindow after X milliseconds
        handler.sendEmptyMessageDelayed(MSG_DISMISS_TOOLTIP, 1500)
    }
    fun isShown(): Boolean {
        return tipWindow?.isShowing ?: false
    }

    fun dismiss() {
        if (tipWindow?.isShowing!!) {
            tipWindow?.dismiss()
        }
    }

    private val handler: Handler = @SuppressLint("HandlerLeak")
    object : Handler() {
        override fun handleMessage(msg: Message) {
            when (msg.what) {
                MSG_DISMISS_TOOLTIP -> if (tipWindow?.isShowing!!) {
                    tipWindow?.dismiss()
                }
            }
        }
    }

    init {
        tipWindow = PopupWindow(ctx)
        inflater = ctx.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        contentView = inflater.inflate(R.layout.popup_frame_item, null)
    }
}