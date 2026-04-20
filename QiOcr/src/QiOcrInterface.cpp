#include <QiOcrInterface.h>
#include "QiOcr.h"

#define QIOCR_VERSION 4

struct QiOcrInterfaceDef : QiOcrInterface
{
	std::vector<std::string> scan_list(const CImage& image, bool skipDet, std::vector<POINT>* centers) override { return ocr->scan_list(image, skipDet, centers); }
	std::vector<std::string> scan_list(const RECT& rect_screen, bool skipDet, std::vector<POINT>* centers) override { return ocr->scan_list(rect_screen, skipDet, centers); }
	std::string scan(const CImage& image, bool skipDet) override { return ocr->scan(image, skipDet); }
	std::string scan(const RECT& rect_screen, bool skipDet) override { return ocr->scan(rect_screen, skipDet); }
	void release() override { if (ocr) { delete ocr; ocr = nullptr; } delete this; }
	bool init(const std::wstring& recFile, const std::wstring& keyFile, const std::wstring& detFile, size_t threads)
	{
		if (ocr) delete ocr;
		ocr = new QiOcrTool(recFile, keyFile, detFile, threads);
		if (ocr)
		{
			if (ocr->isInit()) return true;
			delete ocr;
			ocr = nullptr;
		}
		return false;
	}
	bool init(void* recData, size_t recSize, void* keyData, size_t keySize, void* detData, size_t detSize, size_t threads)
	{
		if (ocr) delete ocr;
		ocr = new QiOcrTool(recData, recSize, keyData, keySize, detData, detSize, threads);
		if (ocr)
		{
			if (ocr->isInit()) return true;
			delete ocr;
			ocr = nullptr;
		}
		return false;
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
QiOcrInterface* _stdcall QiOcrInterfaceInitInterface(const std::wstring& recFile, const std::wstring& keyFile, const std::wstring& detFile, size_t threads)
{
	QiOcrInterfaceDef* ocr = new QiOcrInterfaceDef();
	if (ocr->init(recFile, keyFile, detFile, threads)) return ocr;
	ocr->release();
	return nullptr;
}

#ifdef QIOCR_SHARED
extern "C" __declspec(dllexport)
#endif
QiOcrInterface* _stdcall QiOcrInterfaceInitInterfaceFromMemory(void* recData, size_t recSize, void* keysData, size_t keysSize, void* detData, size_t detSize, size_t threads)
{
	QiOcrInterfaceDef* ocr = new QiOcrInterfaceDef();
	if (ocr->init(recData, recSize, keysData, keysSize, detData, detSize, threads)) return ocr;
	ocr->release();
	return nullptr;
}

#ifndef QIOCR_SHARED
extern "C" size_t _stdcall QiOcrInterfaceVersion()
{
	return QiOcrInterfaceVersionInterface();
}
extern "C" QiOcrInterface* _stdcall QiOcrInterfaceInit(const std::wstring& recFile, const std::wstring& keyFile, const std::wstring& detFile, size_t threads)
{
	return QiOcrInterfaceInitInterface(recFile, keyFile, detFile, threads);
}
extern "C" QiOcrInterface* _stdcall QiOcrInterfaceInitFromMemory(void* recData, size_t recSize, void* keysData, size_t keysSize, void* detData, size_t detSize, size_t threads)
{
	return QiOcrInterfaceInitInterfaceFromMemory(recData, recSize, keysData, keysSize, detData, detSize, threads);
}
#endif