#include "pch.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <regex>

#include <torch/script.h>


#define PA_ASSERTE(condition) if (!(condition)) {throw std::runtime_error("");}


/**
 * Reproduce BertTokenizer https://huggingface.co/transformers/model_doc/bert.html
 */
class Tokenizer
{
public:
	Tokenizer(const std::wstring& vocab_path, bool do_basic_tokenize)
		: vocab(_construct_vocab(vocab_path))
		, do_basic_tokenize(do_basic_tokenize)
	{
		PA_ASSERTE(!do_basic_tokenize);  // TODO
	}

	std::vector<int> encode_sentence(const std::wstring& sentence, int max_len)
	{
		PA_ASSERTE(max_len > 2);  // CLS and SEP
		std::vector<int> res = { CLS_ID };
		std::wregex not_whitespace(LR"(\S+)");
		for (
			auto iter = std::wsregex_iterator(sentence.begin(), sentence.end(), not_whitespace);
			iter != std::wsregex_iterator();
			++iter)
		{
			std::wstring word_not_tokenized = iter->str();
			std::vector<std::wstring> words = _basic_tokenize(word_not_tokenized);
			for (std::wstring word : words)
			{
				while (true)
				{
					auto pos = std::lower_bound(vocab.begin(), vocab.end(), std::make_tuple(word, -1));
					while (true)
					{
						auto[subword, id] = *pos;
						PA_ASSERTE(word.size() >= subword.size());  // TODO remove
						if (subword.size() <= word.size()
							&& word.substr(0, subword.size()) == subword)
						{
							break;
						}
						PA_ASSERTE(pos != vocab.begin());
						--pos;
					}
					auto[subword, id] = *pos;
					res.push_back(id);
					if (res.size() >= max_len - 1)
					{
						res.push_back(SEP_ID);
						return res;
					}
					word = word.substr(subword.size());
					if (word.empty())
					{
						break;
					}
					word = L"##" + word;
				}
			}
		}
		res.push_back(SEP_ID);
		return res;
	}

	std::vector<int> encode(const std::vector<std::wstring>& sentences, int max_text_len)
	{
		std::vector<int> res(max_text_len, 0);
		int curr_pos = 0;
		for (const std::wstring& sent : sentences)
		{
			int max_sent_len = max_text_len - curr_pos;
			std::vector<int> sent_enc = encode_sentence(sent, max_sent_len);
			PA_ASSERTE(sent_enc.size() >= 2 && sent_enc.front() == 101 && sent_enc.back() == 102 && sent_enc.size() <= res.size() - curr_pos);
			std::copy(sent_enc.begin(), sent_enc.end(), res.begin() + size_t(curr_pos));
			curr_pos += sent_enc.size();
			if (curr_pos >= max_text_len - 2)  // -2 for CLS and SEP
			{
				break;
			}
		}
		return res;
	}


private:
	const std::vector<std::tuple<std::wstring, int>> vocab;
	const bool do_basic_tokenize;

	static constexpr auto CLS = L"[CLS]";  // begin of a sentence
	static constexpr auto SEP = L"[SEP]";  // end of a sentence
	static constexpr auto PAD = L"[PAD]";
	static constexpr auto UNK = L"[UNK]";

	static constexpr int CLS_ID = 101;
	static constexpr int SEP_ID = 102;
	static constexpr int PAD_ID = 0;
	static constexpr int UNK_ID = 100;


	static std::vector<std::tuple<std::wstring, int>>
		_construct_vocab(const std::wstring& vocab_path)
	{
		std::wifstream vocab_file(vocab_path);
		PA_ASSERTE(vocab_file.good());
		std::vector<std::tuple<std::wstring, int>> vocab;
		for (std::wstring line; std::getline(vocab_file, line);)
		{
			auto new_item = std::make_tuple(line, int(vocab.size()));
			vocab.push_back(new_item);
		}
		PA_ASSERTE(!vocab.empty());

		if (std::get<0>(vocab.back()).empty())
		{
			vocab.pop_back();
		}
		PA_ASSERTE(!vocab.empty());

		PA_ASSERTE(vocab[size_t(CLS_ID)] == std::make_tuple(CLS, CLS_ID));
		PA_ASSERTE(vocab[size_t(SEP_ID)] == std::make_tuple(SEP, SEP_ID));
		PA_ASSERTE(vocab[size_t(PAD_ID)] == std::make_tuple(PAD, PAD_ID));
		PA_ASSERTE(vocab[size_t(UNK_ID)] == std::make_tuple(UNK, UNK_ID));

		std::sort(vocab.begin(), vocab.end());
		for (auto iter = vocab.begin(); iter != vocab.end(); ++iter)
		{
			PA_ASSERTE(!std::get<0>(*iter).empty());
			PA_ASSERTE(std::next(iter) == vocab.end()
				|| std::get<0>(*iter) != std::get<0>(*std::next(iter)));
		}

		return vocab;
	}

	std::vector<std::wstring>
		_basic_tokenize(const std::wstring& word)
	{
		if (!do_basic_tokenize)
		{
			return { word };
		}
		return { word };  // TODO
	}
};


class TorchscriptModel
{
public:
	TorchscriptModel(const std::string& model_path)
	{
		try {
			model = torch::jit::load(model_path);
		}
		catch (const c10::Error& e) {
			throw std::runtime_error("Error loading the model:\n" + std::string(e.what()) + "\n" + e.what_without_backtrace());
		}
	}

	std::vector<float> infer(const std::vector<int>& token_ids)
	{
		torch::NoGradGuard no_grad;

		PA_ASSERTE(token_ids.size() == 512);
		auto input = torch::tensor(token_ids).unsqueeze(0);
		//std::cout << input << std::endl;

		auto output_l = model.forward({ input }).toList();
		PA_ASSERTE(output_l.size() == 1);
		at::Tensor output = output_l.get(0).toTensor();
		at::Tensor probs = torch::sigmoid(output);
		float* data = probs.data_ptr<float>();
		std::vector<float> res(data, data + probs.size(0));
		return res;
	}

private:
	torch::jit::script::Module model;
};


int main()
{
	//auto tokenizer = Tokenizer(
	//	LR"(C:\Users\alex\PycharmProjects\summarization_nn\data\rus\models\rubert_tokenizer\vocab.txt)",
	//	false
	//);

	//std::wstring line;
	//std::vector<std::wstring> sentences;
	//while (true)
	//{
	//	std::getline(std::wcin, line);
	//	if (line.empty())
	//	{
	//		break;
	//	}
	//	sentences.push_back(line);
	//}

	////sentences = { L"la la la 123 ! la.", L"la la la 123 ! la.", L"Open p.2 of the agreement!" };

	//auto enc = tokenizer.encode(sentences, 512);
	//for (int i : enc)
	//{
	//	std::cout << i << " ";
	//}

	try {
		auto model = TorchscriptModel(
			R"###(C:\Users\alex\PycharmProjects\summarization_nn\data\rus\gazeta\bertsumext_40000_07_10.torchscript)###"
		);
		std::vector<int> token_ids(512, 0);
		std::vector<int> _token_ids = { 
			101,  15393,  26856,    107,    102,    101,  68012,   3032, 105823,
		   1469,  12254,  31888,    851,    877,   8542,  34450,  16988,   5296,
		  23349,    130,    869,  25352,   3989,   1516, 115487,   8388,  44711,
		   1501,    131,    778,  26283,   6093,   3474,    869,  27917,   1455,
		   1469,  13164,  37430,   2241,  37009,    133,    102,    101,   9974,
			133,  11923,   3474,  21368,  44397,   5315,    129,  26133,   5315,
		   5931,  24966,  14026,    157,    102,    101,  15425,   6087,    120,
		  49001,    133,  12254,  31888,    131,   3387,    133,  18655,  13425,
			858,    123,   8980,   2061,  15061,    129,  68913,  47582,    851,
		  56628,  38149,    880,    120,  91717,    133,   9627,   2237,    123,
			133,    102,    101,  29955,   6818,   6379,  26010,  30600,    851,
		  43234,  38149,   1388,    129,   4414,    133,  28512,   5774,  13036,
			133,    102,    101,  11251,  12006,    130,   2061,  28088,    851,
		  97070,    133,    102,    101,   9974,    133,   9627,   2237,   1703,
		  89520,    133,    246,  12039,  13733,    157,    135,    135,  12608,
			275,  28537,    133,  23042,    135,   4228,  19787,    283,    133,
		  14111,  63820,    102,    101,   2988,   8388,  44711,   1501,    131,
			778,  26283,   6093,   1699,  18900,  21958,    129,  65310,    133,
			102,    101,    839,  13989,  32595,   7656,   7241,    898,  25352,
		   3247,   8542,   8528,    133,    102,    101,    815,  14198,  20404,
			845,  12075, 112913,    869,   2789,   9678,   2306,   1997,  25352,
		   3247,  90550,  27245,   2603,  41563,    133,    102,    101,   4665,
		  26756,  16259,  61609,    157, 117289,    129,  23598,   9200,   5935,
		   2604,  73409,    129,  69896,  31226,  55853,    804,  50309,    782,
			832,    788,  49739,  95152,    794,    129, 100273,    880,    135,
		  57884,   9779,    880,    135,   3231,  12018,    135,  45742,   2237,
		  61329,  48662,   2748,  38731,    135,   8009,   2237,    129,   9893,
		   1702,   1916,  75371,    862,    129,  16882,   1714,    129,  18575,
			129,  84432,   1714,    129,  46235,    129,  71558,    120,   5081,
		   1438,  59724,    123,    129, 118701,    862,    129,  22189,  13560,
			135,  34722,   1714,    129,  74028,  45696,    903,    129,  54103,
		   1714,   1641,  47738,    135,  70455,    626,    129,  27665,  10088,
		   2763,   1641,  47738,    135,  70455,    626,    135,  33960,   6014,
			129,  27665,  36728,   2558,  62751,  45647,    129,  27914,   1641,
		  34738,    120,  57884,   1515,    126,  88332,   1503,    123,    129,
		  63530,    626,    120,  76177,  11017,    130,   7974,   8953,  18900,
		  17707,    845,  22620,    123,    129,  16480,  18126,   2068,   4441,
			876,    129,  59364,  17228,   3121,    120,   5640,   3054,    127,
		  20204,    898,    133,  20716,    626,    851,  47989,   1714,    129,
		  43757,   1516,  33887,    123,    129,  29807,  51538,    120,  20027,
		   1465,  22417,  31127,  42380,    123,    129,  34664,  19459,   1641,
		  67534,    851,    135,   2761,  69268,  58736,  18459,   1641,  67534,
			135,  14711,    858,   3565,    129, 115138,    129,  18279,  86151,
			133,    102,    101,  19913,  22584,    130,  14954,   6345,   5296,
		  44336,  21953,   1469,  20447,   3247,    851,   1469,  13037,    129,
		  38795,  12164,   6345,  36592,    129,  74180,    120,   1518,  47738,
			129,   1641,  67534,    123,    851,  16575,   7499,  35332,  89285,
		   1519,    120,  35623,  34082,   3817,    123,    102,    101,  11908,
		  14140,  12832,  16833,    138,   1471,   1703,   3387,    133,  12254,
		  31888,    133,    102,    101,  89143,   1568,  17254,   6675,   7826,
		   1471,   2785,  89228,    102,      0,      0,      0,      0,      0,
			  0,      0,      0,      0,      0,      0,      0,      0,      0,
			  0,      0,      0,      0,      0,      0,      0,      0,      0,
			  0,      0,      0,      0,      0,      0,      0,      0,      0,
			  0,      0,      0,      0,      0,      0,      0,      0,      0,
			  0,      0,      0,      0,      0,      0,      0,      0,      0,
			  0,      0,      0,      0,      0,      0,      0,      0
		};
		std::copy(_token_ids.begin(), _token_ids.end(), token_ids.begin());
		auto res = model.infer(token_ids);
		std::sort(res.begin(), res.end(), std::greater());
		for (float p : res)
		{
			printf("'%.3f', ", std::round(p * 1000) / 1000);
		}
	}
	catch (std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
}



