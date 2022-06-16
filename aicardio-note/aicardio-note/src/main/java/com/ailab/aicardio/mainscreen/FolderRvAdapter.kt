package com.ailab.aicardio.mainscreen

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import android.view.View
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.api.load
import com.ailab.aicardio.R
import com.ailab.aicardio.getBitmapFromDicom
import com.ailab.aicardio.inflate
import com.ailab.aicardio.repository.FolderItem
import com.ailab.aicardio.repository.FolderRepository
import kotlinx.android.extensions.LayoutContainer
import kotlinx.android.synthetic.main.item_folder_horizontal.view.*
import java.io.File
import java.text.SimpleDateFormat

class FolderRvAdapter(private val listener: (View) -> Unit, private val longListener: (View) -> Boolean, private val isVertical: Boolean = false) :
    ListAdapter<FolderItem, FolderRvAdapter.MyViewHolder>(
        NewsItemItemCallback()
    ) {
    companion object {
        const val TAG = "NewsRvAdapter"
        const val FOLDER_ITEM_LIST = "FolderItemList"

        var bitmapFolder : Bitmap? = null
        var bitmapNotDicom: Bitmap? = null

    }


    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): MyViewHolder {
        return MyViewHolder(
            inflate(
                parent.context,
                if (isVertical) R.layout.item_folder_vertical else R.layout.item_folder_horizontal,
                parent
            ), listener, longListener)
    }



    override fun onBindViewHolder(holder: MyViewHolder, position: Int) {
        holder.bind(getItem(position), position)
    }

    private var currentPosition = -1

    override fun getItemCount() = currentList.size

    fun setCurrentPosition(file: String): Int? {

        if (currentPosition != -1) notifyItemChanged(currentPosition)

        for (i in 0..currentList.size - 1) {
//            Log.w(TAG, "setCurrentPosition ${i} ${currentList.get(i)}")

            if(currentList.get(i).path == file) {
                notifyItemChanged(i)
                Log.w(TAG, "setCurrentPosition ${i} ${currentList.get(i)}")
                currentPosition = i
                return i
            }
        }
//        currentPosition = -1
        return null
    }

    fun setCurrentPosition(position: Int) {
        if (currentPosition != -1) notifyItemChanged(currentPosition)
        currentPosition = position
        notifyItemChanged(currentPosition)
    }

//    fun setCurrentPosition(frameIdx: Int) {
//
//    }

//    override fun getItemCount() : Int {
//        Log.w("getItemCount RV FolderList", "${currentList.size}")
//        return currentList.size
//    }

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
//
//        @Suppress("DEPRECATION")
        fun bind(folderItem: FolderItem, position: Int) =
            with(itemView) {
                itemView.tag = folderItem

//                if (position == currentPosition)
                containerView.setBackgroundColor(if (position == currentPosition) containerView.context.getColor(R.color.app_blue_dark) else containerView.context.getColor(R.color.colorBackground))
//                containerView.findViewById<TextView>(R.id.tv_folder_name).text =  basename(fileName) // if (compact) basename(fileName) else fileName


                val simpleDateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm")
//                Logger.getLogger(TAG).warning("${folderItem.name}")

                if (!isVertical) {
                    if (folderItem.name.contains(FolderRepository.DEFAULT_FOLDER_DOWNLOAD)) {
                        tvTitle.text = folderItem.name.subSequence(FolderRepository.DEFAULT_FOLDER_DOWNLOAD.length + 1, folderItem.name.length)

                    } else {
                        tvTitle.text = folderItem.name
                    }
                    val timeFormat = simpleDateFormat.format(folderItem.modifiedTime)

                    tvDescription.text = timeFormat

                } else {
                    tvTitle.text = File(folderItem.name).name
                    tvTitle.textSize  = 10.0F

                    // set background color


                }

                if (bitmapFolder == null) bitmapFolder = BitmapFactory.decodeResource(containerView.resources, R.drawable.file)
                if (bitmapNotDicom == null) bitmapNotDicom = BitmapFactory.decodeResource(containerView.resources, R.drawable.not_dicom_file)

                var bitmap =
                    if( !folderItem.isFile ) bitmapFolder
                    else getBitmapFromDicom(folderItem.path)
                if (bitmap == null)
                    bitmap = bitmapNotDicom


                val hasAnnotation = FolderRepository.getIsAnnotatedFile(folderItem.name)

                Log.w(TAG, "${folderItem.name} ${hasAnnotation}")

                ivHasAnnotation.visibility = if (hasAnnotation) View.VISIBLE else View.GONE

                ivThumbnail.load(bitmap)

            }
    }

    internal class NewsItemItemCallback : DiffUtil.ItemCallback<FolderItem>() {
        override fun areItemsTheSame(oldItem: FolderItem, newItem: FolderItem): Boolean {
            return oldItem == newItem
        }

        override fun areContentsTheSame(oldItem: FolderItem, newItem: FolderItem): Boolean {
            return oldItem.name == newItem.name
        }
    }
}

