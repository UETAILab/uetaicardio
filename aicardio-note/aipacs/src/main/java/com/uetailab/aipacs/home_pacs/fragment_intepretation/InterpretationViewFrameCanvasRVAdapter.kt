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

package com.uetailab.aipacs.home_pacs.fragment_intepretation


import android.content.Context
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.api.load
import com.uetailab.aipacs.R
import kotlinx.android.extensions.LayoutContainer
import kotlinx.android.synthetic.main.item_frame_draw_canvas.view.*

class InterpretationViewFrameCanvasRVAdapter(
    private val listener: (View) -> Unit,
    private val longListener: (View) -> Boolean,
    private val interpretationViewVM: InterpretationViewVM
) :
    ListAdapter<FrameCanvasItem, InterpretationViewFrameCanvasRVAdapter.MyViewHolder>(
        NewsItemItemCallback()
    ) {
    companion object {
        const val TAG = "InterpretationViewFrameCanvasRVAdapter"
    }

    fun inflate(context: Context, viewId: Int, parent: ViewGroup? = null, attachToRoot: Boolean = false): View {
        return LayoutInflater.from(context).inflate(viewId, parent, attachToRoot)
    }

    private var currentPosition = -1

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MyViewHolder {
        return MyViewHolder(
            inflate(parent.context, R.layout.item_frame_draw_canvas, parent), listener, longListener)
    }

    override fun onBindViewHolder(holder: MyViewHolder, position: Int) {
        holder.bind(getItem(position),
            position,
            interpretationViewVM.hasEFBoundaryAndPoint(position),
            interpretationViewVM.hasGLSBoundaryAndPoint(position)
        )
    }

    override fun getItemCount() = currentList.size

    fun setCurrentPosition(frameIdx: Int) {
//        Log.w(TAG, "setCurrentPosition: id: ${frameIdx} cur: ${currentPosition}")
        if (currentPosition != -1 && currentPosition < itemCount) notifyItemChanged(currentPosition)
        currentPosition = frameIdx
        notifyItemChanged(frameIdx)
    }

    fun getCurrentPosition() : Int {
        return currentPosition
    }

    fun updateFrameViewCanvasView(frameIndex: Int) {
        if (frameIndex >= 0 && frameIndex < getItemCount() )notifyItemChanged(frameIndex)
    }

    inner class MyViewHolder(
        override val containerView: View,
        listener: (View) -> Unit,
        longListener: (View) -> Boolean
    ) :
        RecyclerView.ViewHolder(containerView),
        LayoutContainer {

        init {
            itemView.setOnClickListener(listener)
            itemView.setOnLongClickListener(longListener)
        }

        fun bind(frameCanvasItem: FrameCanvasItem, position: Int, hasEFBoundary: Boolean, hasGLSBoundary : Boolean) =
            with(itemView) {
                containerView.setBackgroundColor(if (position == currentPosition) containerView.context.getColor(R.color.greenYellow) else containerView.context.getColor(R.color.white))
                itemView.tag = frameCanvasItem
                iv_frame_canvas_bitmap.load(frameCanvasItem.bitmap)
                tv_frame_canvas_name.setText((frameCanvasItem.index + 1).toString())
                ll_ef_frame_canvas.visibility = if(hasEFBoundary) View.VISIBLE else View.GONE
                ll_gls_frame_canvas.visibility = if(hasGLSBoundary) View.VISIBLE else View.GONE
            }
    }

    internal class NewsItemItemCallback : DiffUtil.ItemCallback<FrameCanvasItem>() {
        override fun areItemsTheSame(oldItem: FrameCanvasItem, newItem: FrameCanvasItem): Boolean {
            return oldItem == newItem
        }

        override fun areContentsTheSame(oldItem: FrameCanvasItem, newItem: FrameCanvasItem): Boolean {
            return oldItem.index == newItem.index
        }
    }


}

