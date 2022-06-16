package com.ailab.aicardio.mainscreen

import android.Manifest
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.viewModels
import androidx.annotation.RequiresApi
import com.ailab.aicardio.R
import com.ailab.aicardio.annotationscreen.AnnotationActivity
import com.ailab.aicardio.checkAndRequestPermissions
import com.ailab.aicardio.repository.FolderItem
import com.ailab.aicardio.toast
import com.google.android.material.snackbar.Snackbar
import com.rohitss.aacmvi.AacMviActivity
import kotlinx.android.synthetic.main.activity_main.*
import java.util.logging.Logger

@RequiresApi(Build.VERSION_CODES.N)
class MainActivity : AacMviActivity<MainViewState, MainViewEffect, MainViewEvent, MainActVM>() {
    override val viewModel: MainActVM by viewModels()

    companion object {
        const val INTENT_FOLDER = "folder"
        const val TAG = "MainActivity"

        fun createIntent(context: Context, folder: String): Intent {
            val intent = Intent(context, MainActivity::class.java)
            intent.putExtra(INTENT_FOLDER, folder)
            return intent
        }



    }

    private val newsRvAdapter by lazy {
        FolderRvAdapter(listener =  {
            viewModel.process(
                MainViewEvent.NewsItemClicked(
                    it.tag as FolderItem, this
                )
            )
        }, longListener = {

            viewModel.process(MainViewEvent.NewsItemLongClicked(it.tag as FolderItem, this))

            Logger.getLogger("TAG").warning("Long Clicked")
            true
        })
    }

    override fun onCreate(savedInstanceState: Bundle?) {

        System.loadLibrary("imebra_lib")
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (checkAndRequestPermissions(this, arrayOf(Manifest.permission.INTERNET, Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE), 1)) {
            intent.getStringExtra(INTENT_FOLDER)?.let {
                Log.w(TAG, "intent folder ${it}")
                viewModel.process(MainViewEvent.FetchNews(it, true))
            } ?: run {
                Log.w(TAG, "intent default folder")
                viewModel.process(MainViewEvent.FetchNews("", true))
            }
        }


        rvNewsHome.adapter = newsRvAdapter

        srlNewsHome.setOnRefreshListener {
            viewModel.process(MainViewEvent.OnSwipeRefresh)
        }
//
//        fabStar.setOnClickListener {
////            viewModel.process(MainViewEvent.FabClicked)
//        }

        btSortName.setOnClickListener {
            viewModel.process(MainViewEvent.SortByName)
        }

        btSortTime.setOnClickListener {
            viewModel.process(MainViewEvent.SortByTime)
        }

    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            1 -> {
                intent.getStringExtra(INTENT_FOLDER)?.let {
                    viewModel.process(MainViewEvent.FetchNews(it, true))
                } ?: run {
                    viewModel.process(MainViewEvent.FetchNews(permissionGranted = true))
                }
            }
        }
    }

    fun setLastFolderAccess() {
        val folder: String? = FolderItem.getFolder(this)
        folder?.let {
            Log.w(TAG, "folder last process: ${folder}")
            val pos = newsRvAdapter.setCurrentPosition(folder)
            Log.w(TAG, "folder position last process: ${pos}")

            pos?.let { rvNewsHome.smoothScrollToPosition(pos) }
        }
    }
    override fun onBackPressed() {

        super.onBackPressed()
//        moveTaskToBack(false)
        Log.w(TAG, "onBackPressed ${FolderItem.getFolder(this)}")
        setLastFolderAccess()

    }
    override fun renderViewState(viewState: MainViewState) {
        when (viewState.mainViewStatus) {
            is MainViewStatus.OpenAnnotateActivity -> {
                val intent = AnnotationActivity.createIntent(
                    this.application,
                    folder = viewState.mainViewStatus.folder,
                    file = viewState.mainViewStatus.file
                )
                startActivity(intent)
            }
            is MainViewStatus.Fetched -> {
                srlNewsHome.isRefreshing = false
                newsRvAdapter.submitList(viewState.folderList)
                setLastFolderAccess()
//                val folder: String? = FolderItem.getFolder(this)
//                folder?.let {
//                    Log.w(TAG, "folder last process: ${folder}")
//                    val pos = newsRvAdapter.setCurrentPosition(folder)
//                    pos?.let { rvNewsHome.smoothScrollToPosition(pos) }
//                }
//
            }
            is MainViewStatus.NotFetched -> {
                Log.w(TAG, "NotFetched ${intent.getStringExtra(INTENT_FOLDER)}")
                intent.getStringExtra(INTENT_FOLDER)?.let {
                    viewModel.process(MainViewEvent.FetchNews(it, false))
                } ?: run {
                    viewModel.process(MainViewEvent.FetchNews("", false))

                }

                srlNewsHome.isRefreshing = false
            }
            is MainViewStatus.Fetching -> {
                srlNewsHome.isRefreshing = true
            }
            is MainViewStatus.OpenMainActivity -> {
                Log.w(TAG, "OpenMainActivity folder = ${viewState.mainViewStatus.folder}")
                val intent = createIntent(this.application, folder = viewState.mainViewStatus.folder)
                val currentPosition = newsRvAdapter.setCurrentPosition(file = viewState.mainViewStatus.folder)

                startActivity(intent)
            }

        }
        tvFolder.text = viewState.folder
    }

    override fun renderViewEffect(viewEffect: MainViewEffect) {
        when (viewEffect) {
            is MainViewEffect.ShowSnackbar -> {
                Snackbar.make(coordinatorLayoutRoot, viewEffect.message, Snackbar.LENGTH_SHORT).show()
            }
            is MainViewEffect.ShowToast -> {
                toast(message = viewEffect.message)
            }
        }
    }
}

