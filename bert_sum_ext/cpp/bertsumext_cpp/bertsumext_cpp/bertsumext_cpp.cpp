#include "pch.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <tuple>
#include <fstream>
#include <regex>
#include <cassert>


#define PA_ASSERTE(condition) assert(condition)


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
	TorchscriptModel(const std::wstring& model_path)
	{

	}

	std::vector<float> infer(const std::vector<int>& token_ids)
	{
		return {};
	}
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

	auto model = TorchscriptModel(
		LR"()"
	);
	std::vector<int> token_ids(512, 0);
	std::vector<int> _token_ids = {101, 23452, 102};
	std::copy(_token_ids.begin(), _token_ids.end(), token_ids.begin());
	auto res = model.infer(token_ids);
	for (float p : res)
	{
		std::cout << p << " ";
	}
}



