#include <QiOcrInterface.h>
#include "QiOcr.h"

#define QIOCR_VERSION 1

struct QiOcrInterfaceDef : QiOcrInterface
{
	std::vector<std::string> scan_list(const CImage& image, bool skipDet = false)
	{
		return ocr->scan_list(image, skipDet);
	}
	std::vector<std::string> scan_list(const RECT& rect_screen, bool skipDet = false)
	{
		return ocr->scan_list(rect_screen, skipDet);
	}
	std::string scan(const CImage& image, bool skipDet = false)
	{
		return ocr->scan(image, skipDet);
	}
	std::string scan(const RECT& rect_screen, bool skipDet = false)
	{
		return ocr->scan(rect_screen, skipDet);
	}
	void release()
	{
		if (ocr)
		{
			delete ocr;
			ocr = nullptr;
		}
	}
	QiOcrInterfaceDef(size_t threads = 0) : ocr(new QiOcrTool(threads))
	{
	}
	QiOcrInterfaceDef(void* recData, size_t recSize, void* keyData, size_t keySize, void* detData, size_t detSize, size_t threads = 0) : ocr(new QiOcrTool(recData, recSize, keyData, keySize, detData, detSize, threads))
	{
	}
	~QiOcrInterfaceDef()
	{
		release();
	}
	QiOcrTool* ocr = nullptr;
};

#ifdef QIOCR_SHARED
extern "C" __declspec(dllexport)
#endif
size_t _stdcall QiOcrInterfaceVersionInterface()
{
	return QIOCR_VERSION;
}

#ifdef QIOCR_SHARED
extern "C" __declspec(dllexport)
#endif
QiOcrInterface* _stdcall QiOcrInterfaceInitInterface(size_t threads)
{
	QiOcrInterfaceDef* ocr = new QiOcrInterfaceDef(threads);
	if (ocr->ocr->isInit()) return (QiOcrInterface*)ocr;
	ocr->release();
	delete ocr;
	return nullptr;
}

#ifdef QIOCR_SHARED
extern "C" __declspec(dllexport)
#endif
QiOcrInterface* _stdcall QiOcrInterfaceInitInterfaceFromMemory(void* recData, size_t recSize, void* keysData, size_t keysSize, void* detData, size_t detSize, size_t threads)
{
	QiOcrInterfaceDef* ocr = new QiOcrInterfaceDef(recData, recSize, keysData, keysSize, detData, detSize, threads);
	if (ocr->ocr->isInit()) return (QiOcrInterface*)ocr;
	ocr->release();
	delete ocr;
	return nullptr;
}

#ifndef QIOCR_SHARED
size_t QiOcrInterfaceVersion()
{
	return QiOcrInterfaceVersionInterface();
}
QiOcrInterface* QiOcrInterfaceInit(size_t threads)
{
	return QiOcrInterfaceInitInterface(threads);
}
QiOcrInterface* QiOcrInterfaceInit(void* recData, size_t recSize, void* keysData, size_t keysSize, void* detData, size_t detSize, size_t threads)
{
	return QiOcrInterfaceInitInterfaceFromMemory(recData, recSize, keysData, keysSize, detData, detSize, threads);
}
#endif