package com.ailab.aicardio.mainscreen

import android.app.Application
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.lifecycle.viewModelScope
import com.rohitss.aacmvi.AacMviViewModel
import com.ailab.aicardio.LCE
import com.ailab.aicardio.repository.FolderItem
import com.ailab.aicardio.repository.FolderRepository
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.launch
import java.io.File

@RequiresApi(Build.VERSION_CODES.N)
class MainActVM(application: Application) :
    AacMviViewModel<MainViewState, MainViewEffect, MainViewEvent>(application) {

    companion object {
        const val TAG = "MainActVM"
    }

    private var count: Int = 0
    private val repository: FolderRepository = FolderRepository.getInstance()
    private var currentSortKey = FolderRepository.SORT_TYPE_NAME_ASC

    init {
        viewState = MainViewState(
            mainViewStatus = MainViewStatus.NotFetched,
            folderList = emptyList(),
            permissionGranted = false,
            folder = FolderRepository.DEFAULT_FOLDER_DOWNLOAD
        )
    }

    override fun process(viewEvent: MainViewEvent) {
        super.process(viewEvent)
        when (viewEvent) {
            is MainViewEvent.NewsItemClicked -> folderItemClicked(viewEvent.folderItem, viewEvent.activity)
            is MainViewEvent.NewsItemLongClicked -> folderItemLongClicked(viewEvent.folderItem, viewEvent.activity)
            MainViewEvent.FabClicked -> fabClicked()
            MainViewEvent.OnSwipeRefresh -> fetchNews(viewState.folder, false)
            is MainViewEvent.FetchNews -> fetchNews(viewEvent.folder, viewEvent.permissionGranted)

            MainViewEvent.PermissionGranted -> {
//                viewState = viewState.copy(permissionGranted = true)
                fetchNews(viewState.folder, permissionGranted=true)
            }
            MainViewEvent.SortByName -> changeSortKeyName()
            MainViewEvent.SortByTime -> changeSortKeyTime()
        }
    }

    private fun changeSortKeyName() {
        when (currentSortKey) {
            FolderRepository.SORT_TYPE_NAME_ASC -> {
                currentSortKey = FolderRepository.SORT_TYPE_NAME_DESC
            }
            else -> {
                currentSortKey = FolderRepository.SORT_TYPE_NAME_ASC
            }
        }
        fetchNews(viewState.folder, false)
    }

    private fun changeSortKeyTime() {
        when (currentSortKey) {
            FolderRepository.SORT_TYPE_TIME_ASC -> {
                currentSortKey = FolderRepository.SORT_TYPE_TIME_DESC
            }
            else -> {
                currentSortKey = FolderRepository.SORT_TYPE_TIME_ASC
            }
        }
        fetchNews(viewState.folder, false)
    }

    private fun folderItemClicked(folderItem: FolderItem, activity: MainActivity) {
        val file = File(folderItem.path)

        if (file.isDirectory) {
//            fetchNews(folder = file.absolutePath)
//            Log.w(TAG, "choose folder ${file.absolutePath}")
            FolderItem.remember(activity, file.absolutePath)

            viewState = viewState.copy(mainViewStatus = MainViewStatus.OpenMainActivity(folder=file.absolutePath ))
        } else {
            FolderItem.remember(activity, file.parentFile!!.absolutePath)
            viewState = viewState.copy(mainViewStatus = MainViewStatus.OpenAnnotateActivity(folder=file.parentFile!!.absolutePath , file = file.absolutePath))
        }

        viewEffect = MainViewEffect.ShowSnackbar(folderItem.name)

    }

    private fun folderItemLongClicked(folderItem: FolderItem, activity: MainActivity) {

        val file = File(folderItem.path)

        if (file.isDirectory) {
            FolderItem.remember(activity, file.absolutePath)
            viewState = viewState.copy(mainViewStatus = MainViewStatus.OpenAnnotateActivity(folder = file.absolutePath, file=null))
        } else {
            FolderItem.remember(activity, file.parentFile!!.absolutePath)
            viewState = viewState.copy(mainViewStatus = MainViewStatus.OpenAnnotateActivity(folder = file.parentFile!!.absolutePath, file=file.absolutePath))
        }

        viewEffect =
            MainViewEffect.ShowSnackbar("Long clicked ${folderItem.name}")

    }


    private fun fabClicked() {
//        count++

//
        val folder = File(viewState.folder).parentFile
        folder?.let {
            if (folder.absolutePath.contains(FolderRepository.DEFAULT_FOLDER_DOWNLOAD)) {
//                viewState = viewState.copy(folder = folder.absolutePath)
                fetchNews(folder.absolutePath, false)
                viewEffect =
                    MainViewEffect.ShowToast(message = "Folder clicked ${folder.absolutePath}")
            } else {
                viewEffect =
                    MainViewEffect.ShowToast(message = "Must be under folder: ${FolderRepository.DEFAULT_FOLDER_DOWNLOAD}")
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.N)
    private fun fetchNews(folderInput: String, permissionGranted: Boolean) {
        Log.w(TAG, "fetchNews folderInput --$folderInput-- ${folderInput.isEmpty()} ${viewState.permissionGranted} && ${permissionGranted}")
        if (!viewState.permissionGranted && !permissionGranted) return
        val folder = if (folderInput.isEmpty()) FolderRepository.DEFAULT_FOLDER_DOWNLOAD else folderInput
        Log.w(TAG, "fetchNews folder --$folder--")

        if (!permissionGranted) {
            viewState = viewState.copy(mainViewStatus = MainViewStatus.Fetching, folder = folder)
        } else {
            viewState = viewState.copy(mainViewStatus = MainViewStatus.Fetching, folder = folder, permissionGranted = true)
        }

        viewModelScope.launch {
            when (val result = repository.getSetLatestFolderList(folder, currentSortKey)) {
                is LCE.Error -> {
                    viewState = viewState.copy(mainViewStatus = MainViewStatus.Fetched)
                    viewEffect =
                        MainViewEffect.ShowToast(
                            message = result.message
                        )
                }
                is LCE.Result -> {
                    viewState =
                        viewState.copy(mainViewStatus = MainViewStatus.Fetched, folderList = result.data)
                }
            }
        }
    }
}